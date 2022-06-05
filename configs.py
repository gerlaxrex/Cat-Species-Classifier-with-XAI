TRAIN_BS = 16
VALID_BS = 1
EPOCHS = 20
IMG_SHAPE = (256, 256)
CLASS_LIST = ['Cheetah', 'Leopard', 'Lion', 'Puma', 'Tiger']
CLASS_DICT = dict(zip(CLASS_LIST, range(len(CLASS_LIST))))
TO_CLASS_DICT = {v: k for k, v in CLASS_DICT.items()}
