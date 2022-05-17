import pandas as pd

class DataCleaner:

    def load_data(self,file_path):
        final_df = pd.read_csv(file_path)
        # class count
        class_count_0, class_count_1 = final_df['label'].value_counts()

        # Separate class
        class_0 = final_df[final_df['label'] == 0]
        class_1 = final_df[final_df['label'] == 1]
        class_0_under = class_1.sample(class_count_1)
        return pd.concat([class_0_under, class_0], axis=0)
