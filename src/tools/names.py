def train_name_creator(args):
    mname = args.model_name
    mname = '-'.join(mname.split('/'))

    additions = ''
    if not args.not_pretrained:
        additions += '_pretrained'
    if args.regularize:
        additions += f'_regularize-{args.reg_type}'
    base_name = f'{mname}_{args.data_name}{additions}_seed{args.seed}'
    return base_name