import os
import cv2
from model_args import segtracker_args,sam_args,aot_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import gc
import imageio
from scipy.ndimage import binary_dilation

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

aot_model2ckpt = {
    "deaotb": "./ckpt/DeAOTB_PRE_YTB_DAV.pth",
    "deaotl": "./ckpt/DeAOTL_PRE_YTB_DAV",
    "r50_deaotl": "./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
}



def tracking_objects_in_video(SegTracker, input_video, input_img_seq, fps):
    
    if input_video is not None:
        return video_type_input_tracking(SegTracker, input_video)
    elif input_img_seq is not None:
        return img_seq_type_input_tracking(SegTracker, input_img_seq, fps)

    return None, None


def video_type_input_tracking(SegTracker, input_video):
    video_name = os.path.basename(input_video).split('.')[0]
    io_args = {
        'input_video': f'{input_video}',
        'output_mask_dir': f'{os.path.join(os.path.dirname(__file__), "assets")}/{video_name}_masks',
        'output_video': f'{os.path.join(os.path.dirname(__file__), "assets")}/{video_name}_seg.mp4', # keep same format as input video
        'output_gif': f'{os.path.join(os.path.dirname(__file__), "assets")}/{video_name}_seg.gif',
    }

    # source video to segment
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # output masks
    output_dir = io_args['output_mask_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pred_list = []
    masked_pred_list = []

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                pred_list.append(SegTracker.refined_merged_mask)
                frame_idx += 1
                continue
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            
            save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx, SegTracker.get_obj_num()),end='\r')
            frame_idx += 1
        cap.release()
        print('\nfinished')
    
    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    cap = cv2.VideoCapture(io_args['input_video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # if io_args['input_video'][-3:]=='mp4':
    #     fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # elif io_args['input_video'][-3:] == 'avi':
    #     fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
    #     # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # else:
    #     fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        masked_pred_list.append(masked_frame)

        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_idx),end='\r')
        frame_idx += 1
    out.release()
    cap.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    print("{} saved".format(io_args['output_gif']))

    # zip predicted mask
    os.system(f"zip -r ./assets/{video_name}_pred_mask.zip {io_args['output_mask_dir']}")

    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()

    return io_args['output_video'], f"./assets/{video_name}_pred_mask.zip"


def img_seq_type_input_tracking(SegTracker, input_img_seq, fps):
    file_name = input_img_seq.name.split('/')[-1].split('.')[0]
    file_path = f'./assets/{file_name}'
    imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])

    video_name = file_name
    io_args = {
        'output_mask_dir': f'{os.path.join(os.path.dirname(__file__), "assets")}/{video_name}_masks',
        'output_video': f'{os.path.join(os.path.dirname(__file__), "assets")}/{video_name}_seg.mp4', # keep same format as input video
        'output_gif': f'{os.path.join(os.path.dirname(__file__), "assets")}/{video_name}_seg.gif',
    }
    
    # output masks
    output_dir = io_args['output_mask_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pred_list = []
    masked_pred_list = []

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0

    with torch.cuda.amp.autocast():
        for img_path in imgs_path:
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                pred_list.append(SegTracker.refined_merged_mask)
                frame_idx += 1
                continue
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            
            save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx, SegTracker.get_obj_num()),end='\r')
            frame_idx += 1
        print('\nfinished')
    
    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    height, width = pred_list[0].shape
    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")

    # if io_args['input_video'][-3:]=='mp4':
    #     fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # elif io_args['input_video'][-3:] == 'avi':
    #     fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
    #     # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # else:
    #     fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    for img_path in imgs_path:
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        masked_pred_list.append(masked_frame)

        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_idx),end='\r')
        frame_idx += 1
    out.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    print("{} saved".format(io_args['output_gif']))

    # zip predicted mask
    os.system(f"zip -r ./assets/{video_name}_pred_mask.zip {io_args['output_mask_dir']}")

    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()


    return io_args['output_video'], f"./assets/{video_name}_pred_mask.zip"