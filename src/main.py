"""
Created on 11/04/2024

@author: Dan
HOW TO RUN: python src/main.py
"""
import os
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    df, _ = pyreadstat.read_sav('data\GBA 7013_Assignment 3 - data.sav')
    # Abusive_Supervision', 'Neuroticism', 'Aggression'
    
    # STEP 1: center Abusive_Supervision and Neuroticism around 0
    df['Abusive_Supervision'] = df['Abusive_Supervision'] - df['Abusive_Supervision'].mean()
    df['Neuroticism'] = df['Neuroticism'] - df['Neuroticism'].mean()

    # STEP 2: Dummy code (No Categorical Variables)
    # STEP 3: Create interaction terms
    df['Interaction'] = df['Abusive_Supervision'] * df['Neuroticism']

    # STEP 4: Hierarchical Regression
    # 4.1 Main Effects
    model_main = smf.ols('Aggression ~ Abusive_Supervision + Neuroticism', data=df).fit()
    # print("Main Effects Regression Summary:")
    # print(model_main.summary())
    
    # 4.2 Interaction Effects
    model_interaction = smf.ols('Aggression ~ Abusive_Supervision + Neuroticism + Interaction', data=df).fit()
    # print("Interaction Effects Regression Summary:")
    # print(model_interaction.summary())

    # STEP 5: Plot the interaction (if significant)
    # Interaction is significant, so we plot the interaction
    Abus_sup_SD = df['Abusive_Supervision'].std()
    Nuerotic_SD = df['Neuroticism'].std()
    print(f"Abus_sup_SD: {round(Abus_sup_SD, 3)}, Nuerotic_SD: {round(Nuerotic_SD, 3)}")
    # Create a grid of values
    Abus_sup = df['Abusive_Supervision'].mean() + Abus_sup_SD * 1.5 * (2 * (0.5 - 0.5))
    Nuerotic = df['Neuroticism'].mean() + Nuerotic_SD * 1.5 * (2 * (0.5 - 0.5))
    print(f"Abus_sup: {round(Abus_sup, 3)}, Nuerotic: {round(Nuerotic, 3)}")
    # Create a grid of values
    x = pd.DataFrame({'Abusive_Supervision': [Abus_sup]*100, 'Neuroticism': [Nuerotic]*100})
    x['Abusive_Supervision'] = x['Abusive_Supervision'] + Abus_sup_SD * 1.5 * (2 * (pd.Series(range(100))/100 - 0.5))
    x['Neuroticism'] = x['Neuroticism'] + Nuerotic_SD * 1.5 * (2 * (pd.Series(range(100))/100 - 0.5))
    # Predict the values
    y = model_interaction.predict(x)
    # Plot the interaction
    import matplotlib.pyplot as plt
    plt.plot(x['Abusive_Supervision'], y, label='Neuroticism')
    plt.xlabel('Abusive Supervision')
    plt.ylabel('Aggression')
    plt.title('Interaction Plot')
    plt.legend()
    plt.show()
    # save figure to ./figures/interaction_plot.png
    # enusre path exists
    os.makedirs('figures', exist_ok=True)

    plt.savefig('figures/interaction_plot.png')





if __name__ == '__main__':
    main()