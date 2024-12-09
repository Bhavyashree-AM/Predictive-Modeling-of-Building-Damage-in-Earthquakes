def run_classifier():
    
    loading_window = ttk.Toplevel(window)
    screen_width = loading_window.winfo_screenwidth()
    screen_height = loading_window.winfo_screenheight()

    # Calculate the x and y coordinates to center the window
    x = (screen_width / 2) - (400 / 2)
    y = (screen_height / 2) - (200 / 2)

# Set the window's position
    loading_window.geometry(f"400x200+{int(x)}+{int(y)}")
    loading_window.title('Loading...')
    loading_label = ttk.Label(loading_window, text='Running classifier...', font=('Helvetica', 16))
    loading_label.pack(padx=50,pady=70)
    
    def classifier():
    
        complete_unprocessed_values = pd.read_csv('values.txt',on_bad_lines='skip')
        complete_labels = pd.read_csv('labels.txt')

        train_unprocessed_values, test_unprocessed_values, train_labels, test_labels = train_test_split(complete_unprocessed_values, complete_labels, test_size = 0.1, stratify=complete_labels.iloc[:,1])

        train_unprocessed_values, validation_unprocessed_values, train_labels, validation_labels = train_test_split(train_unprocessed_values, train_labels, test_size = 0.1, stratify=train_labels.iloc[:,1])
    
        train_unprocessd_values_labels = train_unprocessed_values.merge(train_labels, how = 'left', on = 'building_id')

        selected_features = ['has_superstructure_mud_mortar_stone',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_adobe_mud',
            'geo_level_1_id', 'age', 'has_superstructure_stone_flag',
            'foundation_type', 'roof_type',
            'has_superstructure_cement_mortar_brick', 'count_floors_pre_eq',
            'other_floor_type', 'count_families', 'has_superstructure_timber',
            'geo_level_2_id', 'has_superstructure_rc_engineered',
            'has_superstructure_bamboo', 'position',
            'has_superstructure_rc_non_engineered', 'ground_floor_type']

        train_selected_features = train_unprocessed_values[selected_features]
        test_selected_features = test_unprocessed_values[selected_features]
        validation_selected_features =  validation_unprocessed_values[selected_features]

        train_labels = train_labels['damage_grade']
        test_labels = test_labels['damage_grade']
        validation_labels = validation_labels['damage_grade']

        # removing 'foundation_type' feature
        train_selected_features.drop(columns = ['foundation_type'], inplace = True)
        test_selected_features.drop(columns = ['foundation_type'], inplace = True)
        validation_selected_features.drop(columns = ['foundation_type'], inplace = True)

        # transforming the age

        train_selected_features['age_log'] = np.log(train_selected_features['age'] + (0.1))
        train_selected_features.drop(columns = ['age'], inplace = True)

        validation_selected_features['age_log'] = np.log(validation_selected_features['age'] + (0.1))
        validation_selected_features.drop(columns = ['age'], inplace = True)

        test_selected_features['age_log'] = np.log(test_selected_features['age'] + (0.1))
        test_selected_features.drop(columns = ['age'], inplace = True)

        categorical_variables = ['roof_type', 'position', 'ground_floor_type', 'other_floor_type']

        # Use pd.get_dummies() to one-hot encode the categorical features
        train_selected_features_encoded = pd.get_dummies(train_selected_features, columns=categorical_variables)
        test_selected_features_encoded = pd.get_dummies(test_selected_features, columns=categorical_variables)
        validation_selected_features_encoded = pd.get_dummies(validation_selected_features, columns=categorical_variables)

        scaler = MinMaxScaler()

        cols_to_scale = ['geo_level_1_id', 'count_floors_pre_eq', 'count_families', 'geo_level_2_id', 'age_log']

        train_selected_features_encoded[cols_to_scale] = scaler.fit_transform(train_selected_features_encoded[cols_to_scale])
        test_selected_features_encoded[cols_to_scale] = scaler.transform(test_selected_features_encoded[cols_to_scale])
        validation_selected_features_encoded[cols_to_scale] = scaler.transform(validation_selected_features_encoded[cols_to_scale])

        sm = SMOTE(random_state=42)
        train_values_os, train_labels_os = sm.fit_resample(train_selected_features_encoded, train_labels)

        le = LabelEncoder()
        all_categorical_columns = ['land_surface_condition', 'foundation_type', 'roof_type',
            'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']

        for col in all_categorical_columns:
            train_unprocessed_values[col] = le.fit_transform(train_unprocessed_values[col])
            test_unprocessed_values[col] = le.transform(test_unprocessed_values[col])
            validation_unprocessed_values[col] = le.transform(validation_unprocessed_values[col])

        pca = PCA(n_components = 17)
        PCA_train = pca.fit_transform(train_values_os)
        PCA_val = pca.fit_transform(validation_selected_features_encoded)
        PCA_test = pca.fit_transform(test_selected_features_encoded)
        model.fit(train_values_os, train_labels_os - 1)

        xgb_processed_test_predictions = model.predict(test_selected_features_encoded)

        accuracy= accuracy_score(test_labels-1,xgb_processed_test_predictions)
        f1= f1_score(test_labels-1,xgb_processed_test_predictions,average='weighted')

        pred_table= pd.DataFrame()
        pred_table["Building ID"]= test_selected_features_encoded.index
        pred_table["Predicted Damage Grade"]= xgb_processed_test_predictions+1
        pred_table["Actual Damage Grade"] = (test_labels).to_numpy()

        accuracy_label = ttk.Label(window, text='Accuracy Score: ', font=('Arial', 14))
        accuracy_label.pack(pady=5)
        f1_label = ttk.Label(window, text='F1 Score: ', font=('Arial', 14))
        f1_label.pack(pady=5)

        table_frame = ttk.Frame(window)
        table_frame.pack(pady=5)
        table_label = ttk.Label(table_frame, text='Prediction Table', font=('Arial', 13))
        table_label.pack(side=ttk.TOP)
        table = ttk.Treeview(table_frame, columns=('Building ID', 'Predicted Damage Grade','Actual Damage Grade'), show='headings')
        table.column("Building ID", width=200, anchor="center")
        table.column("Predicted Damage Grade", width=200, anchor="center")
        table.column("Actual Damage Grade", width=200, anchor="center")
        table.heading('Building ID', text='Building ID')
        table.heading('Predicted Damage Grade', text='Predicted Damage Grade')
        table.heading('Actual Damage Grade', text='Actual Damage Grade')
        table.pack()
        loading_window.destroy()
        print(accuracy_label.config(text='Accuracy Score: {:.2f}'.format(accuracy)))
        print(f1_label.config(text='F1 Score: {:.2f}'.format(f1)))

        # Clear the table and insert the new rows
        for i in table.get_children():
            table.delete(i)
        for index, row in pred_table.iterrows():
            table.insert('', index, values=(row['Building ID'], row['Predicted Damage Grade'],row['Actual Damage Grade']))

    window.after(100,classifier)

if __name__ == '__main__':

    import ttkbootstrap as ttk
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    with open('xgb_model.pickle', 'rb') as f:
        model = pickle.load(f)

    window= ttk.Window(themename='darkly')
    window.title('Classification Model')
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the x and y coordinates to center the window
    x = (screen_width / 2) - (600 / 2)
    y = (screen_height / 2) - (400 / 2)

    # Set the window's position
    window.geometry(f"600x400+{int(x)}+{int(y)}")

    label = ttk.Label(window,text= 'XGBoost Classifier',background='white',foreground='black',borderwidth=4)
    label.pack(pady=10)

    x_input=ttk.Label(window, text='Input File: values.txt',background='white',foreground='black',borderwidth=3)
    x_input.pack(side= ttk.TOP,padx=5,pady=5)

    y_input= ttk.Label(window, text='Target File: labels.txt',background='white',foreground='black',borderwidth=3)
    y_input.pack(side= ttk.TOP,pady=5)

    run_button = ttk.Button(window, text="Run Classifier", style="success.TButton",command=run_classifier)
    run_button.pack(pady=10)

    window.mainloop()