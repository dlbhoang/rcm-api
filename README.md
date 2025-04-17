# **Sentiment Analysis in ShopeeFood & Build Streamlit App for End User**
![Image](https://huynhthaibao.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F35e3b43e-ce00-46c4-a295-07a0dee3bd1a%2Fe5a92fa3-e103-4cf2-9c01-3fa2f474918a%2F87e66f79-6c91-4e06-ba75-e28bb110696c.png?table=block&id=6ba26cc2-3260-4d78-81a0-13ac5a806a62&spaceId=35e3b43e-ce00-46c4-a295-07a0dee3bd1a&width=1390&userId=&cache=v2)



## **🔍Project Overview**  

- We are always excited to explore new things every day, and cuisine is a field that attracts a lot of attention. When choosing a new restaurant, we tend to consider reviews from those who have enjoyed it to decide whether or not to try it.  

- This is becoming increasingly important in the foodservice industry. Restaurants need to strive to improve the quality of their food and service attitude in order to maintain their reputation and attract new customers.  

- This project is based on data from 3,615 restaurants with over 15,000 reviews collected from the online food delivery platform ShopeeFood in order to build a system to support restaurants in classifying customer feedback into 3 groups: positive, negative, and neutral through text data.  

- You can view the full source code for data collection and download the dataset at the [Website](https://www.notion.so/1f0cd452570f44548f9f6796b6bffc14?pvs=21).  



## **📌Business Goal**

- Data normalization of Vietnamese text for model building.  

- Utilize Machine Learning algorithms to categorize customer feedback into three groups: Positive, Negative, and Neutral.  

- Construct an end-user application that enables users to generate new sentiment predictions/classifications based on trained Machine Learning data.  



## **🗂Work Scope**

- Data collection from the ShopeeFood platform using the Selenium library through [Project Crawl Data.](https://www.notion.so/1f0cd452570f44548f9f6796b6bffc14?pvs=21)  

- Analysis of the dataset overview.  

- Employing Machine Learning algorithms like Decision Tree, XGBoots, Random Forest, etc., to classify customer sentiment.  

- Developing an end-user application using Streamlit.  



## **💡Domain Knowledge**

#### **What is Sentiment Analysis?**  
Sentiment analysis (also known as opinion mining, emotion analysis, and sentiment mining) is the field of using natural language processing, text analysis, computational linguistics, and biometrics to identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to documents such as reviews and survey responses, social media, news media, and other online content for applications ranging from marketing to customer relationship management and clinical medicine.  
![Image](https://huynhthaibao.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F35e3b43e-ce00-46c4-a295-07a0dee3bd1a%2F05f63558-3261-45a7-91ed-c900f34dc8e0%2F93ca72b4-58f0-4396-8128-1d91c5d8c035.png?table=block&id=b3b76d84-6637-4dd9-95fa-644f21e5a893&spaceId=35e3b43e-ce00-46c4-a295-07a0dee3bd1a&width=1440&userId=&cache=v2)  
Sentiment Analysis is the process of analyzing and evaluating a person's opinion about a particular object (positive, negative, or neutral opinion, etc.). This process can be carried out using rule-based methods, Machine Learning, or a hybrid approach (combining the two methods).  
Sentiment Analysis is widely applied in practice, especially in business promotion activities. By analyzing user reviews of a product to see whether they are negative, positive, or highlight the product's limitations, companies can improve product quality, enhance their company's image, and strengthen customer satisfaction.
[Read more…](https://monkeylearn.com/sentiment-analysis/)

#### **What is Streamlit?**  
Streamlit is a tool specifically designed for Machine Learning Engineers to create simple, user-friendly web interfaces for end-users. It is a complete product with high applicability.  
![Image](https://huynhthaibao.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F35e3b43e-ce00-46c4-a295-07a0dee3bd1a%2F89017762-676c-465a-ac38-ca3ad8cdadc9%2FUntitled.png?table=block&id=e0938825-fd00-470d-b9c8-b2c2891bacee&spaceId=35e3b43e-ce00-46c4-a295-07a0dee3bd1a&width=2000&userId=&cache=v2)  
Streamlit is known as the fastest way to turn your code into an application product, "The fastest way to build and share data apps."  
With its rich set of built-in functions that enhance app interactivity, Streamlit is the ideal tool for those who want to build a complete product.  
[Read more…](https://streamlit.io/)



## **💻Technologies and Tools**  

- Clean data: Using Python’s libraries like Pandas and Numpy. 

- Machine Learning: Using Python’s library Sklearn likes: Decision Tree, Random Forest, XGBoots,...  

- Analysis: Using Python’s libraries like Scipy, Pandas and Numpy.  

- Visualization: Creating informative visualizations using Python’s libraries like Matplotlib, Seaborn and Plotly.  

- User’s App: Using Streamlit to build and deloy app for end user.   



## **🛠Work Reference**  

### **I. Data Preparation**  

Using file: [Data_Preparation.ipynb](https://github.com/baothaihuynh/ShopeeFood_SentimentAnalysis/blob/master/Data_Preparation.ipynb)

### **II. Data Exploratory Analysis**    

Using file: [Data_Analysis.ipynb](https://github.com/baothaihuynh/ShopeeFood_SentimentAnalysis/blob/master/Data_Analysis.ipynb)

- Task 1. Trend of restaurant/eatery distribution by district    

- Task 2. Customer Comment Trends by Region  

- Task 3. Operating Hour Trends of Restaurants/Eateries on the ShopeeFood Platform    

- Task 4. Food Price Trends    

- Task 5. Overview of User Review Categorization  

- Task 6. Number of Reviews by Restaurant  

- Task 7. Restaurants with 100 Reviews  

- Task 8. Restaurants with Fewer than 5 Reviews  

- Task 9. Conclusion  

### **III. Sentiment Analysis Using Machine Learning**  

Using file: [Data_MachineLearning.ipynb](https://github.com/baothaihuynh/ShopeeFood_SentimentAnalysis/blob/master/Data_MachineLearning.ipynb) 

- Task 1. Read Summary Dataset  

- Task 2. Build Models using Cross_Validate with DecisionTreeClassifier, RandomForestClassifier, XGBootsClassifier, NaiveBayes  

- Task 3. Build XGBoots Model  

- Task 4. Using Over Sampling Data  

- Task 5. Save Model  

- Task 6. Test Samples  

### **IV. Deloy App for End-Users by Streamlit**  

Using repo: [SentimentAnalysis_Deloy_Streamlit](https://github.com/baothaihuynh/SentimentAnalysis_Deloy_Streamlit)

![Image](https://huynhthaibao.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F35e3b43e-ce00-46c4-a295-07a0dee3bd1a%2F6de025af-39c5-4125-8175-3886ece21d1d%2FUntitled.png?table=block&id=724d2cc1-4ef8-4b5f-9368-d539cef745e5&spaceId=35e3b43e-ce00-46c4-a295-07a0dee3bd1a&width=2000&userId=&cache=v2)  

You can access and use the Sentiment Analysis App at https://sentimentanalysis-deloy.streamlit.app/

#### `💭Thank you for reading!!!`
