from decouple import config, UndefinedValueError


class Config:
    def __init__(self):
        try:
            self.test_flag = config("TEST_FLAG", cast=bool)
            self.train_flag = config("TRAIN_FLAG", cast=bool)
            self.full_data_flag = config("FULL_DATA_FLAG", cast=bool)
            self.n_th_frame = config("N_TH_FRAME", cast=bool)
            self.prev_f = config("PREV_FRAMES", cast=int)
            self.future_f = config("FUTURE_FRAMES", cast=int)
            self.start_f = config("START_FUTURE", cast=int)
            self.DRR = config("DATA_REDUCTION_RATE", cast=int)
            self.model_name = config("MODEL_NAME")
            self.do_frame_averaging = config("DO_FRAME_AVERAGING")
            self.dataset_type = config("DATASET_TYPE")
            self.dataset_path = config("DATASET_PATH")

        except UndefinedValueError as e:
            raise ValueError(f"Environment variable {e} is not set.")
