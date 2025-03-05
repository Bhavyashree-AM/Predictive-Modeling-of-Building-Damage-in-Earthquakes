# Earthquake Damage Grade Prediction

## Overview
This project aims to predict the damage grades of buildings impacted by earthquakes using machine learning models. The goal is to provide valuable insights for urban planners, engineers, and disaster management agencies to improve earthquake resilience and mitigate risks. Various machine learning algorithms, including Logistic Regression, Random Forests, Neural Networks, and XGBoost, were explored for accurate predictions of damage grades.

## GUI Interface
We have developed a user-friendly graphical user interface (GUI) using the `tkinter` library. The GUI incorporates the powerful XGBoost algorithm, which was the top-performing model in our project. The GUI allows users to compare predicted damage grades with actual damage grades by inputting corresponding feature values.

Additionally, the GUI supports flexible data input, accepting both `.csv` and `.txt` file formats. Users can easily upload datasets in either format for damage grade prediction.

## Conclusion
The project's primary aim is to predict the damage grades of buildings affected by earthquakes, providing insights to urban planners, engineers, and disaster management agencies. Using machine learning algorithms, including Logistic Regression, Random Forests, Neural Networks, and XGBoost, we explored their potential for accurate damage grade predictions. The data preprocessing steps ensured the models could handle missing values and encoded categorical values.

The models were evaluated using the micro-averaged F1 score, offering insights for better decision-making and resource allocation. The findings from this study have several key implications:

1. **Policy and Planning:** Accurate damage predictions can inform urban planners, engineers, and policymakers in developing earthquake-resistant infrastructure and city planning strategies, prioritizing high-risk areas, and implementing effective mitigation measures.

2. **Early Warning Systems:** Integrating predictive models into early warning systems can allow authorities to issue timely alerts, enabling populations in earthquake-prone regions to take necessary precautions, minimizing loss of life and property.

3. **Construction Guidelines:** The study can contribute to robust construction guidelines that ensure buildings are designed to withstand seismic forces, improving resilience in earthquake-prone regions.

## Future Directions
With an additional 3-6 months of development, we plan to explore the potential of deep neural networks to further enhance our model’s performance. While deep neural networks require significant computational power and large training datasets, they offer promising opportunities to improve accuracy. To optimize the benefits of this approach, we will collaborate with domain experts to identify key features for precise predictions. This collaboration will ensure our model is not only accurate but also practically relevant for real-world applications.

We will also fine-tune the neural network architecture and training process, experimenting with different structures and hyperparameters to identify the most effective approach. Ultimately, adopting deep neural networks will require assessing available computational resources, training data, and collaborating with experts to ensure the model’s relevance and accuracy.

