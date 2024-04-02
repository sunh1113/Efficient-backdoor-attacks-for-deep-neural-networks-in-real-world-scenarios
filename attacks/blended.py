from PIL import Image


class Blended(object):
    def __init__(self, img_size, num, triggers, mode=0, target=0):
        super(Blended, self).__init__()
        self.img_size = img_size
        self.num = num
        self.mode = mode
        self.target = target
        self.trigger = Image.open('./triggers/0.jpg').resize((self.img_size, self.img_size), Image.BILINEAR)

    def __call__(self, img, target, backdoor, idx):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if (self.mode == 0 and idx >= self.num) or self.mode == 2:
            target, backdoor = self.target, 1
            img = Image.blend(img, self.trigger, 0.15)
        return img, target, backdoor

    def set_mode(self, mode):
        self.mode = mode
