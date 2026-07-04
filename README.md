Floods are among the most devastating natural disasters, claiming thousands of lives and displacing millions every year. Despite their recurring nature, the lack of timely and accurate early-warning systems continues to amplify their destructive impact. Conventional forecasting methods often fall short in predicting floods at the right time, leaving authorities and communities with insufficient opportunity to respond.

This project addresses that gap by building a machine learning-powered flood prediction system trained on historical weather data. Using classification algorithms Decision Tree, Random Forest, K-Nearest Neighbours (KNN), and XGBoost the system analyses meteorological features such as annual rainfall, cloud visibility, and seasonal rainfall patterns to predict the likelihood of a flood event.

The best-performing model is saved and integrated into a Flask web application, enabling concerned authorities and disaster management teams to monitor flood risk predictions through an intuitive and accessible user interface. The application is designed for potential deployment on IBM Cloud, making it scalable and accessible from any location.

Scenario 1: Early Flood Warning and Evacuation Planning

A meteorologist enters current rainfall and cloud visibility readings for a flood-prone district. The model analyses the inputs and predicts a high probability of flooding, allowing authorities to issue evacuation advisories several hours in advance.

Scenario 2: Disaster Response and Resource Allocation

A disaster relief coordinator uses the web application during monsoon season to monitor multiple regions simultaneously. By entering regional weather data for each area, the system provides instant flood risk classifications, helping prioritise resource deployment.

Scenario 3: Model Validation and Performance Assessment

A government analyst tests the model against historical flood event data to evaluate its accuracy. The XGBoost model achieves 96.55% accuracy on test data, confirming the system’s reliability for operational use.

**Skills Required**
  1. Machine Learning Algorithms
  
  2. NumPy (Python Package)
  
  3. Matplotlib (Python Package)
  
  4. Scikit-Learn (Python Package)
  
  5. Supervised Learning
  
  6. Flask (Web Framework)
  
  7. K-Nearest Neighbors Algorithm
  
  8. K-Means Clustering
  
  9. PyTorch (Machine Learning Library)

**Entity Relationship Diagram**

  <img width="1536" height="1024" alt="ER Model" src="https://github.com/user-attachments/assets/4dc7e4f7-1aa4-458d-a43c-b488191a24ea" />

**Pre-requisites**

  The essential software, tools, and knowledge required to successfully develop and run the project.

  i. Anaconda Navigator: A desktop graphical user interface used to launch applications and manage Python packages and environments for data science. 
      
      https://www.anaconda.com/download

  ii. PyCharm: An Integrated Development Environment (IDE) specifically designed for Python programming, offering code analysis and debugging tools. 
    
      https://www.jetbrains.com/pycharm/

  iii. NumPy: Core library for numerical computing in Python, providing support for large, multi-dimensional arrays and mathematical functions. 
  
      https://numpy.org/doc/stable/

  iV. Pandas: Data manipulation and analysis library offering data structures and operations for manipulating numerical tables and time series. 
  
      https://pandas.pydata.org/docs/

  V. Scikit-learn: Machine learning library featuring classification, regression, and clustering algorithms like support vector machines and random forests. 
      
      https://scikit-learn.org/stable/

  Vi. Matplotlib: Comprehensive library for creating static, animated, and interactive visualizations in Python. 
  
      https://matplotlib.org/stable/

  Vii. Seaborn: Data visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics. 
      
      https://seaborn.pydata.org/

  Vii. Flask: A lightweight WSGI web application framework used to build web APIs and deploy machine learning models. 
  
      https://flask.palletsprojects.com/

**The main process starts from the below onwards**

  **1. Data Collection:**

         https://www.kaggle.com/datasets/arbethi/rainfall-dataset

**2. Visualizing and Analysing the Data**

  i. Importing the Libraries

  <img width="242" height="72" alt="image" src="https://github.com/user-attachments/assets/c9364d8b-9905-4ade-8be8-867e53672f59" />
 
  ii. Reading the Dsataset

  <img width="654" height="264" alt="image" src="https://github.com/user-attachments/assets/650fb310-b755-4017-b850-65c577c7495d" />

  iii. Univariate Analysis
  
  <img width="722" height="532" alt="image" src="https://github.com/user-attachments/assets/5d30ab5a-b9be-41d0-bc76-2afce84aed7e" />

  iV. Multivariate Analysis
  
  <img width="654" height="617" alt="image" src="https://github.com/user-attachments/assets/0395e308-b4b6-4241-b2d3-424478d1c0fd" />
  
  V. Descriptive Analysis
  
  <img width="581" height="478" alt="image" src="https://github.com/user-attachments/assets/d7ceefe6-574e-4882-801d-092d7f3d6f91" />

  <img width="764" height="473" alt="image" src="https://github.com/user-attachments/assets/44397750-7614-4f7f-acb1-2372002912a1" />

**3. Data Pre-processing**

  i. Handling Missing Values

  <img width="218" height="559" alt="image" src="https://github.com/user-attachments/assets/2860ae33-9d6b-4579-ac42-08a87adf724f" />

  ii. Handling Outliers

  a. Visualize Outliers
  
  <img width="862" height="490" alt="image" src="https://github.com/user-attachments/assets/510b2eb9-984b-4be0-a9b6-11888d62c57f" />

  b. IQR Method

  <img width="168" height="267" alt="image" src="https://github.com/user-attachments/assets/89b13013-8233-4814-a29a-0c78571e3e90" />

  c. Count Outliers

  <img width="254" height="246" alt="image" src="https://github.com/user-attachments/assets/4d1d98b2-a786-4dbf-827d-9cd2ca5857ca" />
  
  iii. Handling Categorical Values

  <img width="144" height="200" alt="image" src="https://github.com/user-attachments/assets/515b74c2-5bfe-45ae-aafd-a1ae2d24555f" />

  iV. Splitting Data into Training and Test Sets

  <img width="542" height="377" alt="image" src="https://github.com/user-attachments/assets/42af5089-e561-486e-a8e6-baea515985e1" />

  <img width="312" height="264" alt="image" src="https://github.com/user-attachments/assets/b2a685fb-57c0-45e8-8058-d45a30656e5d" />

  V. Feature Scaling

  <img width="307" height="624" alt="image" src="https://github.com/user-attachments/assets/f946697f-4dc9-484a-bea7-55de5bf96feb" />

  <img width="274" height="591" alt="image" src="https://github.com/user-attachments/assets/8d2fff94-4822-4d80-a3a8-7246963525c5" />

  <img width="277" height="377" alt="image" src="https://github.com/user-attachments/assets/c9c809f9-e671-4d01-8879-f2beeca5ff8c" />
  
**4. Model Building**

  i. Decision Tree Model

  <img width="259" height="203" alt="image" src="https://github.com/user-attachments/assets/9fbaa3ff-7576-4fb6-b577-35e5edf8fa0e" />

  <img width="350" height="391" alt="image" src="https://github.com/user-attachments/assets/e340de5d-8f7c-427d-91ff-0bdf8ea788e9" />

  <img width="313" height="493" alt="image" src="https://github.com/user-attachments/assets/e2a6bea9-e37b-4066-a9a3-0c9833d7d333" />

  <img width="357" height="411" alt="image" src="https://github.com/user-attachments/assets/ea74cefa-a83c-4307-a8c9-4ded76789de5" />

  ii. Random Forest Model

  <img width="490" height="525" alt="image" src="https://github.com/user-attachments/assets/15876725-8942-42f6-be48-b49325145f2b" />

  <img width="355" height="307" alt="image" src="https://github.com/user-attachments/assets/bd60f7c3-d4bb-47b3-abff-97928ba73e5b" />

  <img width="270" height="100" alt="image" src="https://github.com/user-attachments/assets/76683f31-a8c4-4b6d-b1c2-b023930b8bda" />

  iii. KNN Model

  <img width="481" height="475" alt="image" src="https://github.com/user-attachments/assets/7c6a6b56-b38d-4459-b62e-5909be58e120" />

  <img width="319" height="483" alt="image" src="https://github.com/user-attachments/assets/3f0a5a83-9f7e-4f6d-9a6f-b6d322aed64c" />

  iV. XGBoost Model

  <img width="481" height="521" alt="image" src="https://github.com/user-attachments/assets/87519d97-6643-45e0-8f5a-c4415fad81f5" />

  <img width="373" height="305" alt="image" src="https://github.com/user-attachments/assets/ba737d72-cfcd-4a0f-80b3-d3975de5b351" />

  V. Comparing the Models

  <img width="282" height="512" alt="image" src="https://github.com/user-attachments/assets/82f0d5ed-c0b1-4efd-8a1c-32c7c804d737" />

  Vi. Evaluating Performance and Saving the Model

  <img width="552" height="585" alt="image" src="https://github.com/user-attachments/assets/7ad1e7c3-a1e8-4771-988a-4740de469dfa" />

  <img width="270" height="113" alt="image" src="https://github.com/user-attachments/assets/949425f8-2f7c-4a36-a7e6-43e36d49ba71" />

**5. Application Building**

  i. Building HTML Pages

  <img width="1365" height="722" alt="image" src="https://github.com/user-attachments/assets/646dff5b-06ee-4cbc-999e-6a76f7c3d55d" />

  <img width="608" height="494" alt="image" src="https://github.com/user-attachments/assets/d24e2fa3-a1d9-4e5f-968b-d8fa6ef0cee2" />

  ii. Build the Python Script (app.py)
  
  <img width="1365" height="717" alt="image" src="https://github.com/user-attachments/assets/f5577ec8-c81a-4568-b959-d5f520a670aa" />

  iii. Run the Application

  <img width="353" height="244" alt="image" src="https://github.com/user-attachments/assets/cc31772d-fc4a-42d3-9b00-c515f2a4df3e" />

Rising Waters focuses on addressing the growing risk of floods, which have become increasingly frequent and severe due to factors such as climate change, unpredictable weather patterns, rapid urbanization, deforestation, and inadequate drainage and water management systems. Floods can have devastating consequences, including loss of human life, displacement of communities, destruction of homes and infrastructure, agricultural damage, economic losses, and long-term environmental impacts. As populations continue to grow and weather conditions become more extreme, the need for accurate flood prediction and effective disaster preparedness has become more critical than ever.

This project aims to leverage the power of data analysis and machine learning to develop a flood prediction system capable of identifying potential flood risks before they occur. By analyzing historical weather data, rainfall patterns, water levels, and other relevant environmental factors, the system can detect trends and generate predictions that support early warning mechanisms. These insights enable government agencies, disaster management authorities, and local communities to take proactive measures, such as issuing alerts, planning evacuations, and implementing preventive strategies.

In addition to improving disaster response, flood prediction systems contribute to better resource allocation, infrastructure planning, and environmental management. Early identification of high-risk areas helps reduce damage, minimize economic losses, and improve public safety. The integration of intelligent prediction models into disaster management frameworks can significantly enhance preparedness and resilience against flood-related events.

Ultimately, Rising Waters demonstrates how data-driven technologies can be used to tackle real-world environmental challenges. By providing timely and reliable flood risk assessments, the project supports informed decision-making and helps protect lives, property, and ecosystems from the increasing threats posed by flooding in a changing world.

**To run the website **
  
  <img width="1109" height="350" alt="image" src="https://github.com/user-attachments/assets/fb752b41-41a1-4545-b5e7-6b0f30475ed3" />

  1.Home Page:

  <img width="1365" height="483" alt="image" src="https://github.com/user-attachments/assets/8be6e0da-8c26-4a7e-8ac1-88ca4dce3ae7" />

  **giving values from the Flood Dataset.excel**
  
  2. Flood Prediction page:

  <img width="1365" height="672" alt="image" src="https://github.com/user-attachments/assets/72b68ba2-e9c4-480e-a9c4-8bf50d4c76ef" />

  3. Testing

  i. Safe Zone
  
  <img width="1365" height="672" alt="image" src="https://github.com/user-attachments/assets/033496da-732d-4eea-8ffd-5c9e678d5457" />
  
  **Result**
  
  <img width="476" height="270" alt="image" src="https://github.com/user-attachments/assets/6dd76745-9358-4a2f-b343-87c1e16e184a" />

  ii. Flood alert
