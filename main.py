import numpy as np
import pandas as pd

class FairnessComputation:
    def __init__(self, num_seekers, num_categories):
        self.num_seekers = num_seekers
        self.num_categories = num_categories
        self.SV = np.zeros((num_seekers, num_categories))  # sensitive values (one-hot encoded)
       # self.RS = np.random.rand(num_seekers, num_categories)  This process will store the RS which we don't want to store
        
        self.V_TTR = np.zeros((num_seekers, num_categories))  # for TTP
        self.V_MO = np.zeros((num_seekers, num_categories))  # for MO

    def data_donation(self, SV):
            self.SV = SV

    
    
    def generate_and_compute_V_values(self):
            """
            Generate random secrets (RS), calculate V_TTR and V_MO using RS, 
            and delete RS after computation to ensure it is not stored.
            """
            RS = np.random.rand(self.num_seekers, self.num_categories)  # Generate random secrets
            self.V_TTR = self.SV + RS
            self.V_MO = self.SV - RS
            del RS  # Delete RS after using it

    
    
    def reconstruct_SV(self):
            """
            Reconstruct the sensitive values using both V_TTR and V_MO.
            """
            reconstructed_SV = (self.V_TTR + self.V_MO) / 2  # Average to cancel out RS
            return reconstructed_SV
        
    
    
    
    def calculate_sum_ttr_mo(self):
            """
            Calculate the summation of V_TTR and V_MO across all seekers and categories.
            """
            sum_ttr = np.sum(self.V_TTR)
            sum_mo = np.sum(self.V_MO)
            return sum_ttr, sum_mo

    
    
    
    def calculate_diversity(self):
            """
            Calculate the diversity (DIV) using the formula: (sum_ttr + sum_mo) / N
            where N is the total number of job seekers.
            """
            sum_ttr, sum_mo = self.calculate_sum_ttr_mo()
            diversity = (sum_ttr + sum_mo) / (2 * self.num_seekers)  # Since both sums are additive
            return diversity

    
    
    
    def calculate_expectation(self, weights):
            """
            Calculate the weighted expectation using the sensitive values (SV) and weights.
            """
            expectation = 0
            for i in range(self.num_seekers):
                for k in range(self.num_categories):
                    if self.SV[i, k] == 1:  # If the SV is in this category
                        expectation += weights[k]
            return expectation

def load_sensitive_values(file_path):
    """
    Load sensitive values from a CSV file.
    """
    return pd.read_csv(file_path, header=None).values

# Example usage:

# Load sensitive values from CSV file
file_path = 'sensitve_values.csv'
SV = load_sensitive_values(file_path)

num_seekers, num_categories = SV.shape

    # Initialize fairness computation class
fairness = FairnessComputation(num_seekers, num_categories)

    # Data donation (input sensitive values)
fairness.data_donation(SV)

    # Generate V_TTR and V_MO values and delete the random secrets after use
fairness.generate_and_compute_V_values()

    # Reconstruct the original sensitive values from TTP data
reconstructed_SV = fairness.reconstruct_SV()
print("Reconstructed Sensitive Values (SV):")
print(reconstructed_SV)

    # Calculate diversity based on V_TTR and V_MO
diversity = fairness.calculate_diversity()
print(f"Diversity: {diversity}")
    # Define some weights for expectation calculation
    
weights = np.array([1.0, 2.0, 3.0])  # Importance of categories

    # Calculate expectation using the weights
expectation = fairness.calculate_expectation(weights)
print(f"Expectation (EXP): {expectation}")