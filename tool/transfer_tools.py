import cv2
import numpy as np 

def mask2bbox(mask):
    if len(np.where(mask > 0)[0]) == 0:
        print(f'not mask')
        return np.array([[0, 0], [0, 0]]).astype(np.int64)

    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)

    x0 = np.min(np.nonzero(x_)[0])
    x1 = np.max(np.nonzero(x_)[0])
    y0 = np.min(np.nonzero(y_)[0])
    y1 = np.max(np.nonzero(y_)[0])

    return np.array([[x0, y0], [x1, y1]]).astype(np.int64)

if __name__ == '__main__':
    mask = cv2.imread('./debug/painter_input_mask.jpg', -1)[2:, 2:]
    bbox = mask2bbox(mask)
    draw_0 = cv2.rectangle(mask, bbox[0], bbox[1], (0, 0, 255))
    cv2.imwrite('./debug/rect.png', draw_0)