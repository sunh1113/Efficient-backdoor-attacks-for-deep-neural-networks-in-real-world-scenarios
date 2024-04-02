from attacks.blended import Blended
from attacks.badnets import Badnets
from attacks.optimized import Optimized


TRIGGERS = {
    'blend': Blended,
    'bad': Badnets,
    'uap': Optimized,
    'cfa': Optimized,
}

def build_trigger(attack_name, img_size, num, trigger, mode, target):
    assert attack_name in TRIGGERS.keys()
    trigger = TRIGGERS[attack_name](img_size, num, trigger, mode, target)
    return trigger
