# -*- coding: utf-8 -*-
import argparse
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import scipy.ndimage
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from . import RESULTS_DIR, WEIGHTS_DIR
from .core.utils import to_tensors
from .model.misc import get_device
from .model.modules.flow_comp_raft import RAFT_bi
from .model.propainter import InpaintGenerator
from .model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .utils.download_util import load_file_from_url

warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_VIDEO = PACKAGE_ROOT / 'inputs' / 'object_removal' / 'bmx-trees'
DEFAULT_MASK_DIR = PACKAGE_ROOT / 'inputs' / 'object_removal' / 'bmx-trees_mask'
DEFAULT_OUTPUT_ROOT = RESULTS_DIR


@dataclass
class InpaintingConfig:
    """Configuration for ProPainter video inpainting.
    
    Attributes:
        resize_ratio: Resize scale for processing video (default: 1.0).
        height: Height of the processing video (-1 means use original).
        width: Width of the processing video (-1 means use original).
        mask_dilation: Mask dilation for video and flow masking (default: 4).
        ref_stride: Stride of global reference frames (default: 10).
        neighbor_length: Length of local neighboring frames (default: 10).
        subvideo_length: Length of sub-video for long video inference (default: 80).
        raft_iter: Iterations for RAFT inference (default: 20).
        fp16: Use fp16 (half precision) during inference (default: False).
        device: Device to run inference on (default: auto-detect).
    """
    resize_ratio: float = 1.0
    height: int = -1
    width: int = -1
    mask_dilation: int = 4
    ref_stride: int = 10
    neighbor_length: int = 10
    subvideo_length: int = 80
    raft_iter: int = 20
    fp16: bool = False
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = get_device()


class ProPainterModel:
    """ProPainter model wrapper for video inpainting.
    
    This class loads and caches the models for efficient repeated inference.
    
    Example:
        >>> model = ProPainterModel()
        >>> frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        >>> masks = np.zeros((10, 480, 640), dtype=np.uint8)
        >>> masks[:, 100:200, 100:200] = 255  # Region to inpaint
        >>> result = model.inpaint(frames, masks)
    """
    
    def __init__(self, device: Optional[torch.device] = None, fp16: bool = False):
        """Initialize ProPainter models.
        
        Args:
            device: Device to run inference on (default: auto-detect).
            fp16: Use fp16 (half precision) during inference (default: False).
        """
        self.device = device if device is not None else get_device()
        self.fp16 = fp16
        if self.device == torch.device('cpu'):
            self.fp16 = False
            
        self._fix_raft = None
        self._fix_flow_complete = None
        self._model = None
        
    def _load_models(self):
        """Lazy load models on first use."""
        if self._model is not None:
            return
            
        # Load RAFT
        ckpt_path = load_file_from_url(
            url=os.path.join(pretrain_model_url, 'raft-things.pth'),
            model_dir=str(WEIGHTS_DIR), progress=True, file_name=None
        )
        self._fix_raft = RAFT_bi(ckpt_path, self.device)
        
        # Load flow completion network
        ckpt_path = load_file_from_url(
            url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
            model_dir=str(WEIGHTS_DIR), progress=True, file_name=None
        )
        self._fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in self._fix_flow_complete.parameters():
            p.requires_grad = False
        self._fix_flow_complete.to(self.device)
        self._fix_flow_complete.eval()
        
        # Load ProPainter model
        ckpt_path = load_file_from_url(
            url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
            model_dir=str(WEIGHTS_DIR), progress=True, file_name=None
        )
        self._model = InpaintGenerator(model_path=ckpt_path).to(self.device)
        self._model.eval()
        
    def inpaint(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        config: Optional[InpaintingConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """Inpaint video frames using ProPainter.
        
        Args:
            frames: Input video frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
            masks: Masks indicating regions to inpaint, shape (T, H, W) or (T, H, W, 1).
                   Values > 0 indicate regions to inpaint. Can be uint8 or bool.
            config: Inpainting configuration. If None, uses default settings.
            progress_callback: Optional callback function(current, total) for progress updates.
            
        Returns:
            Inpainted frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
        """
        if config is None:
            config = InpaintingConfig(device=self.device, fp16=self.fp16)
            
        self._load_models()
        
        return _inpaint_numpy(
            frames=frames,
            masks=masks,
            fix_raft=self._fix_raft,
            fix_flow_complete=self._fix_flow_complete,
            model=self._model,
            config=config,
            progress_callback=progress_callback
        )


def _prepare_frames_from_numpy(
    frames: np.ndarray,
    size: Optional[Tuple[int, int]] = None
) -> Tuple[List[Image.Image], Tuple[int, int], Tuple[int, int]]:
    """Convert numpy frames to PIL Images and resize if needed.
    
    Args:
        frames: Input frames as numpy array of shape (T, H, W, 3) with dtype uint8.
        size: Optional target size as (width, height).
        
    Returns:
        Tuple of (list of PIL Images, process_size, out_size).
    """
    pil_frames = [Image.fromarray(f) for f in frames]
    return resize_frames(pil_frames, size)


def _prepare_masks_from_numpy(
    masks: np.ndarray,
    length: int,
    size: Tuple[int, int],
    flow_mask_dilates: int = 8,
    mask_dilates: int = 5
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Process numpy masks for inpainting.
    
    Args:
        masks: Masks as numpy array of shape (T, H, W) or (T, H, W, 1).
        length: Number of frames.
        size: Target size as (width, height).
        flow_mask_dilates: Dilation iterations for flow masks.
        mask_dilates: Dilation iterations for inpainting masks.
        
    Returns:
        Tuple of (flow_masks, masks_dilated) as lists of PIL Images.
    """
    # Ensure masks are 2D per frame
    if masks.ndim == 4:
        masks = masks.squeeze(-1)
    
    # Normalize to 0-255 uint8
    if masks.dtype == bool:
        masks = masks.astype(np.uint8) * 255
    elif masks.max() <= 1:
        masks = (masks * 255).astype(np.uint8)
    
    flow_masks = []
    masks_dilated = []
    
    for i in range(masks.shape[0]):
        mask_img = Image.fromarray(masks[i]).convert('L')
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_arr = np.array(mask_img)
        
        # Dilate for flow mask
        if flow_mask_dilates > 0:
            flow_mask_arr = scipy.ndimage.binary_dilation(
                mask_arr, iterations=flow_mask_dilates
            ).astype(np.uint8)
        else:
            flow_mask_arr = binary_mask(mask_arr.copy()).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_arr * 255))
        
        # Dilate for inpainting mask
        if mask_dilates > 0:
            mask_arr = scipy.ndimage.binary_dilation(
                mask_arr, iterations=mask_dilates
            ).astype(np.uint8)
        else:
            mask_arr = binary_mask(mask_arr).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_arr * 255))
    
    # If single mask provided, repeat for all frames
    if len(flow_masks) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length
        
    return flow_masks, masks_dilated


def _inpaint_numpy(
    frames: np.ndarray,
    masks: np.ndarray,
    fix_raft: RAFT_bi,
    fix_flow_complete: RecurrentFlowCompleteNet,
    model: InpaintGenerator,
    config: InpaintingConfig,
    progress_callback: Optional[callable] = None
) -> np.ndarray:
    """Core inpainting function that works with numpy arrays.
    
    Args:
        frames: Input frames as numpy array of shape (T, H, W, 3).
        masks: Masks as numpy array of shape (T, H, W) or (T, H, W, 1).
        fix_raft: RAFT optical flow model.
        fix_flow_complete: Flow completion network.
        model: ProPainter inpainting model.
        config: Inpainting configuration.
        progress_callback: Optional progress callback.
        
    Returns:
        Inpainted frames as numpy array.
    """
    device = config.device
    use_half = config.fp16 and device != torch.device('cpu')
    
    # Determine target size
    original_size = (frames.shape[2], frames.shape[1])  # (W, H)
    if config.width != -1 and config.height != -1:
        size = (config.width, config.height)
    elif config.resize_ratio != 1.0:
        size = (int(config.resize_ratio * original_size[0]), 
                int(config.resize_ratio * original_size[1]))
    else:
        size = None
    
    # Prepare frames
    pil_frames, process_size, out_size = _prepare_frames_from_numpy(frames, size)
    w, h = process_size
    
    # Prepare masks
    flow_masks, masks_dilated = _prepare_masks_from_numpy(
        masks, len(pil_frames), process_size,
        flow_mask_dilates=config.mask_dilation,
        mask_dilates=config.mask_dilation
    )
    
    # Convert to tensors
    frames_inp = [np.array(f).astype(np.uint8) for f in pil_frames]
    frames_tensor = to_tensors()(pil_frames).unsqueeze(0) * 2 - 1
    flow_masks_tensor = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated_tensor = to_tensors()(masks_dilated).unsqueeze(0)
    
    frames_tensor = frames_tensor.to(device)
    flow_masks_tensor = flow_masks_tensor.to(device)
    masks_dilated_tensor = masks_dilated_tensor.to(device)
    
    video_length = frames_tensor.size(1)
    
    with torch.no_grad():
        # Compute optical flow
        if frames_tensor.size(-1) <= 640:
            short_clip_len = 12
        elif frames_tensor.size(-1) <= 720:
            short_clip_len = 8
        elif frames_tensor.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
            
        if frames_tensor.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames_tensor[:, f:end_f], iters=config.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames_tensor[:, f-1:end_f], iters=config.raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames_tensor, iters=config.raft_iter)
            torch.cuda.empty_cache()
            
        if use_half:
            frames_tensor = frames_tensor.half()
            flow_masks_tensor = flow_masks_tensor.half()
            masks_dilated_tensor = masks_dilated_tensor.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()
            
        # Complete flow
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > config.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, config.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + config.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + config.subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks_tensor[:, s_f:e_f+1]
                )
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks_tensor[:, s_f:e_f+1]
                )
                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_tensor)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_tensor)
            torch.cuda.empty_cache()
            
        # Image propagation
        masked_frames = frames_tensor * (1 - masks_dilated_tensor)
        subvideo_length_img_prop = min(100, config.subvideo_length)
        
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                
                b, t, _, _, _ = masks_dilated_tensor[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f],
                    pred_flows_bi_sub,
                    masks_dilated_tensor[:, s_f:e_f],
                    'nearest'
                )
                updated_frames_sub = frames_tensor[:, s_f:e_f] * (1 - masks_dilated_tensor[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated_tensor[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated_tensor.size()
            prop_imgs, updated_local_masks = model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated_tensor, 'nearest'
            )
            updated_frames = frames_tensor * (1 - masks_dilated_tensor) + \
                            prop_imgs.view(b, t, 3, h, w) * masks_dilated_tensor
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            
    ori_frames = frames_inp
    comp_frames = [None] * video_length
    
    neighbor_stride = config.neighbor_length // 2
    if video_length > config.subvideo_length:
        ref_num = config.subvideo_length // config.ref_stride
    else:
        ref_num = -1
        
    # Feature propagation + transformer
    total_steps = (video_length + neighbor_stride - 1) // neighbor_stride
    current_step = 0
    
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                            min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, config.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated_tensor[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :]
        )
        
        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, 
                           selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks_np = masks_dilated_tensor[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
                
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks_np[i] \
                    + ori_frames[idx] * (1 - binary_masks_np[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + \
                                      img.astype(np.float32) * 0.5
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
                
        torch.cuda.empty_cache()
        
        current_step += 1
        if progress_callback is not None:
            progress_callback(current_step, total_steps)
            
    # Resize to original size if needed
    if out_size != process_size:
        comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
        
    # Stack into numpy array
    result = np.stack(comp_frames, axis=0)
    
    torch.cuda.empty_cache()
    
    return result


def inpaint(
    frames: np.ndarray,
    masks: np.ndarray,
    config: Optional[InpaintingConfig] = None,
    progress_callback: Optional[callable] = None
) -> np.ndarray:
    """Inpaint video frames using ProPainter.
    
    This is a convenience function that creates a ProPainterModel instance
    and runs inpainting. For repeated calls, consider using ProPainterModel
    directly to avoid reloading models.
    
    Args:
        frames: Input video frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
        masks: Masks indicating regions to inpaint, shape (T, H, W) or (T, H, W, 1).
               Values > 0 indicate regions to inpaint. Can be uint8 or bool.
        config: Inpainting configuration. If None, uses default settings.
        progress_callback: Optional callback function(current, total) for progress updates.
        
    Returns:
        Inpainted frames as numpy array of shape (T, H, W, 3) with dtype uint8 (RGB).
        
    Example:
        >>> import numpy as np
        >>> from propainter import inpaint, InpaintingConfig
        >>> 
        >>> # Load your video frames (T, H, W, 3) RGB uint8
        >>> frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        >>> 
        >>> # Create masks (T, H, W) - non-zero values indicate regions to inpaint
        >>> masks = np.zeros((10, 480, 640), dtype=np.uint8)
        >>> masks[:, 100:200, 100:200] = 255  # Region to inpaint
        >>> 
        >>> # Inpaint with default settings
        >>> result = inpaint(frames, masks)
        >>> 
        >>> # Or with custom config
        >>> config = InpaintingConfig(fp16=True, mask_dilation=8)
        >>> result = inpaint(frames, masks, config=config)
    """
    if config is None:
        config = InpaintingConfig()
        
    model = ProPainterModel(device=config.device, fp16=config.fp16)
    return model.inpaint(frames, masks, config=config, progress_callback=progress_callback)


def _to_path(path_like):
    return Path(path_like)

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        file_path = Path(file_path)
        dir_name = file_path.parent.resolve()
        dir_name.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(file_path), img, params)


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size


#  read frames from video
def read_frame_from_videos(frame_root):
    frame_root = str(frame_root)
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    mpath = str(mpath)
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

def main(argv=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default=str(DEFAULT_INPUT_VIDEO), help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default=str(DEFAULT_MASK_DIR), help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default=str(DEFAULT_OUTPUT_ROOT), help='Output folder. Default: package results directory')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--no-save-frames', action='store_true', help='Disable saving output frames.')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

    args = parser.parse_args(argv)

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if args.fp16 else False 
    if device == torch.device('cpu'):
        use_half = False

    video_path = _to_path(args.video)
    mask_path = _to_path(args.mask)
    output_root = _to_path(args.output)

    frames, fps, size, video_name = read_frame_from_videos(video_path)
    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)
    
    fps = args.save_fps if fps is None else fps
    save_root = output_root / video_name
    save_root.mkdir(parents=True, exist_ok=True)

    if args.mode == 'video_inpainting':
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(mask_path, frames_len, size, 
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError
    
    # for saving the masked frames or video
    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]),2).repeat(3, axis=2)/255.
        img = np.array(frames[i])
        green = np.zeros([h, w, 3]) 
        green[:,:,1] = 255
        alpha = 0.6
        # alpha = 1.0
        fuse_img = (1-alpha)*img + alpha*green
        fuse_img = mask_ * fuse_img + (1-mask_)*img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

    
    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir=str(WEIGHTS_DIR), progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir=str(WEIGHTS_DIR), progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()


    ##############################################
    # set up ProPainter model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir=str(WEIGHTS_DIR), progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    
    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')
    with torch.no_grad():
        # ---- compute flow ----
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        # use fp32 for RAFT
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=args.raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
            torch.cuda.empty_cache()


        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        
        # ---- complete flow ----
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, args.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()
            

        # ---- image propagation ----
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, args.subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                       pred_flows_bi_sub, 
                                                                       masks_dilated[:, s_f:e_f], 
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            
    
    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1
    
    # ---- feature propagation + transformer ----
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
        
        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            
            # pred_img = selected_imgs # results of image propagation
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        
        torch.cuda.empty_cache()
                
    # save each frame
    save_frames = not args.no_save_frames

    if save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = save_root / 'frames' / f'{idx:04d}.png'
            imwrite(f, img_save_root)
                    

    # if args.mode == 'video_outpainting':
    #     comp_frames = [i[10:-10,10:-10] for i in comp_frames]
    #     masked_frame_for_save = [i[10:-10,10:-10] for i in masked_frame_for_save]
    
    # save videos frame
    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    
    print(f'\nAll results are saved in {save_root}')
    
    torch.cuda.empty_cache()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())