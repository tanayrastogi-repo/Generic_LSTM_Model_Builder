import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as sc
import math

def file_selector(folder_path = ""):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    return os.path.join(folder_path, selected_filename)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def LSTMmodel(input_layer, look_back):	
    model = Sequential()
    model.add(LSTM(input_layer, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def main():
    st.title("ML Pipeline") 

    activities = ["Exploratory Data Analysis", "Plotting and Visualization", "Building Model", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    # ----------- File Selection ---------------"
    if choice != "About":   
        st.subheader("File Selection")
        # Find paths of the dataset
        path = os.path.join(os.getcwd(), 'datasets')
        filename = file_selector(path)
        st.info("You selected {}".format(filename))

        # Read dataset
        df = pd.read_csv(filename)


    if choice == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        # Show Dataset
        if st.checkbox("Show Dataset"):
            number = st.number_input("Number of Rows to view", 5, len(df))
            st.dataframe(df.head(number))
        # Show Columns
        if st.checkbox("Column Names"):
            st.write(df.columns)   
        # Show Shape
        if st.checkbox("Shape of Dataset"):
            data_dim = st.radio("Show Dimensions by", ("Rows", "Columns"))
            if data_dim == "Columns":
                st.text("Number of Columns: ")
                st.write(df.shape[1])
            elif data_dim == "Rows":
                st.text("Number of Rows: ")
                st.write(df.shape[0])
            else:
                st.write(df.shape)
        # Select columns
        if st.checkbox("Columns to show"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
        # Show Values
        if st.checkbox("Show 'Target' value count"):
            st.text("Value count by Target/Class")
            st.write(df.iloc[:, -1].value_counts())
        # Show Datatypes
        if st.checkbox("Show Datatypes"):
            st.text("Datatypes by columns")
            st.write(df.dtypes)
        # Show Data summary
        if st.checkbox("Show Data Summary"):
            st.text("Datatypes Summary")
            st.write(df.describe().T)



    elif choice == "Plotting and Visualization":
        st.subheader("Plotting and Visualization")
        all_columns = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "line", "scatter", "pie", "bar", "correlation"]) 
        
        if type_of_plot=="line":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.line_chart(cust_data)
        
        elif type_of_plot=="area":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.area_chart(cust_data)  
        
        elif type_of_plot=="bar":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.bar_chart(cust_data)
        
        elif type_of_plot=="pie":
            select_columns_to_plot = st.selectbox("Select a column", all_columns)
            st.write(df[select_columns_to_plot].value_counts().plot.pie())
            st.pyplot()
        
        elif type_of_plot=="correlation":
            st.write(sns.heatmap(df.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}))
            st.pyplot()

        elif type_of_plot=="scatter":
            st.write("Scatter Plot")
            scatter_x = st.selectbox("Select a column for X Axis", all_columns)
            scatter_y = st.selectbox("Select a column for Y Axis", all_columns)
            st.write(sns.scatterplot(x=scatter_x, y=scatter_y, data = df))
            st.pyplot()
        


    elif choice == "Building Model":
        
        st.write("Select the columns to use for training")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select Columns", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

        st.subheader("Scaling of Dataset")
        type_of_scaler = st.selectbox("Select Type of Scaler", ["StandardScaller", "MinMax"]) 
        if type_of_scaler == "StandardScaller":
            scaler = sc.StandardScaler()
            new_df = new_df.astype('float32')
            new_df = scaler.fit_transform(new_df) 
        elif type_of_scaler == "MinMax":
            scaler = sc.MinMaxScaler(feature_range=(0, 1))
            new_df = new_df.astype('float32')
            new_df = scaler.fit_transform(new_df)

        st.write("Scaled Dataset")
        number = st.number_input("Number of Rows to view", 5, len(new_df))
        st.dataframe(new_df[:number])
        
        st.subheader("Train/Test Split")
        split_ratio = st.slider("Train dataset ratio", min_value = float(0), max_value = float(1), step=float(0.01))
        train_size = int(len(new_df) * float(split_ratio))
        train, test = new_df[0:train_size,:], new_df[train_size:len(new_df),:]
        st.write("Length of Training Data: ", len(train))
        st.write("length of Test Data: ", len(test))
        if st.checkbox("Show Dataset"):
            st.write("Training data")    
            st.dataframe(train)
            st.write("Test data")
            st.dataframe(test)

        st.subheader("Reshape dataset for Modelling")	
        # reshape into X=t and Y=t+1
        look_back = st.number_input("Number of Timesteps to lookback", 1, len(new_df))
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        st.subheader("Training Model")
        model = None
        model_selection = st.selectbox("Model to train", ["Select Model", "LSTM", "MLP", "RNN"])
        if model_selection == "LSTM":
            n_neuron = st.slider("Number of LSTM Neurons", min_value = int(1), max_value = int(100), step=int(1))
            n_epochs = st.slider("Number of epochs", min_value = int(1), max_value = int(150), step=int(1))
            model = LSTMmodel(input_layer=n_neuron, look_back=look_back)
            
        if model_selection == "MLP":
            pass    
        if model_selection == "RNN":
            pass
        
        if st.button("Start Training"):
            model.fit(trainX, trainY, epochs=n_epochs, batch_size=1, verbose=2)

            st.subheader("Predictions")	
            if model != None:
                trainPredict = model.predict(trainX)
                testPredict = model.predict(testX)
                # invert predictions
                trainPredict = scaler.inverse_transform(trainPredict)
                trainY = scaler.inverse_transform([trainY])
                testPredict = scaler.inverse_transform(testPredict)
                testY = scaler.inverse_transform([testY])
                # calculate root mean squared error
                trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                st.write('Train Score: %.2f RMSE' % (trainScore))
                testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                st.write('Test Score: %.2f RMSE' % (testScore))

                # Plots
                trainPredictPlot = np.empty_like(new_df)
                trainPredictPlot[:, :] = np.nan
                trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
                # shift test predictions for plotting
                testPredictPlot = np.empty_like(new_df)
                testPredictPlot[:, :] = np.nan
                testPredictPlot[len(trainPredict)+(look_back*2)+1:len(new_df)-1, :] = testPredict
                # plot baseline and predictions
                plot_data = pd.DataFrame(np.hstack((scaler.inverse_transform(new_df), trainPredictPlot, testPredictPlot)), columns = ["Data", "Training", "Test"])
                st.line_chart(plot_data)
        
        
            




    else:
        st.subheader("About")
        st.write("The app is made by Tanay Rastogi.")
        st.write("References")
        st.write("MachineLearningMastery   : https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/")
        st.write("JCharisTech & J-Secur1ty : https://www.youtube.com/playlist?list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU")
 



if __name__ == "__main__":
    main()
    



