# Data analysis Project - Gender Disparities in Employment and Unemployment Across Danish Industries
This project aims to analyze the gross unemployment rate in Denmark across different industries, focusing on identifying patterns and trends in the labor market that can inform policy decisions and support efforts to reduce unemployment in the country. Specifically, we will explore how the gender gap in unemployment rates varies across different industries and whether certain industries are more prone to higher levels of unemployment than others.

The project presents the analysis as a "story" to provide a reader-friendly flow. Rather than importing all the necessary data at once and explaining it in detail, the project visualizes the area of interest and imports relevant data as the analysis progresses. This approach allows readers to follow along more easily and better understand how the analysis was conducted.

The entire analysis and results are presented in the dataproject.ipynb file.

## Applied Datasets
The project utilizes the following datasets pulled from **Statistics Denmark** using DstAPI:
1. AULP01 - Gross Unemplyment by Gender
2. NAN1 - Real year-to-year change in GDP
3. RAS300 - Employment by Industry

## Applied Dependencies
To replicate our work, apart from a standard Anaconda Python 3 installation, the project requires installation of **DstAPI**. To install DstAPI, follow these steps:
1. Open the Anaconda prompt.
2. Run the command 'pip install git+https://github.com/alemartinello dstapi' in the prompt.
3. For a more detailed guide on using the DST API, please refer to https://github.com/alemartinello/dstapi.

## Replecation of Analysis
To replecate our work, you can simply:

1. Clone the project repository
2. Install the necessary dependencies
3. Run the **dataproject.ipynb** file to reproduce the analysis