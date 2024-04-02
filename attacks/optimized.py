import numpy as np
from PIL import Image


class Optimized(object):
    def __init__(self, img_size, num, triggers, mode=0, target=0):
        super(Optimized, self).__init__()
        self.img_size = img_size
        self.num = num
        self.mode = mode
        self.target = target
        self.trigger = (triggers * 255)[0, :, :, :].transpose(1, 2, 0)

    def __call__(self, img, target, backdoor, idx):
        if (self.mode == 0 and idx >= self.num) or (self.mode == 2):
            target, backdoor = self.target, 1
            img = np.array(img)
            img = np.clip(img + self.trigger, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        return img, target, backdoor

    def set_mode(self, mode):
        self.mode = mode
