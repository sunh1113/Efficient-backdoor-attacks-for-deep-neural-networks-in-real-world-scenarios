
def get_name(opts):
    name = 'error'

    if opts.constraint == 'numbers':
        name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
            opts.data_name,
            opts.constraint,
            opts.poison_source,
            opts.attack_name,
            opts.model_name,
            opts.poison_ratio,
            opts.attack_target,
            opts.suffix
        )
    elif opts.constraint == 'classes':
        name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            opts.data_name,
            opts.constraint,
            opts.poison_source,
            opts.attack_name,
            opts.model_name,
            opts.poison_ratio,
            opts.class_nums,
            opts.class_idx,
            opts.attack_target,
            opts.suffix
        )
    elif opts.constraint == 'domains':
        name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
            opts.data_name,
            opts.constraint,
            opts.poison_source,
            opts.attack_name,
            opts.model_name,
            opts.poison_ratio,
            opts.attack_target,
            opts.suffix
        )
    else:
        name = name

    return name