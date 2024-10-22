import numpy as np

class FairnessComputation:
    def __init__(self, num_seekers, num_categories):
        self.num_seekers = num_seekers
        self.num_categories = num_categories
        self.SV = np.zeros((num_seekers, num_categories))  # sensitive values (one-hot encoded)
        self.RS = np.random.rand(num_seekers, num_categories)  # random secrets
        
        self.V_TTR = np.zeros((num_seekers, num_categories))  # for TTP
        self.V_MO = np.zeros((num_seekers, num_categories))  # for MO

    def data_donation(self, SV):
        """
        Store sensitive values (SV) from job seekers, encoded in one-hot format.
        """
        self.SV = SV

    def generate_V_values(self):
        """
        Generate V_TTR and V_MO by adding random secrets to the sensitive values.
        """
        for i in range(self.num_seekers):
            self.V_TTR[i] = self.SV[i] + self.RS[i]
            self.V_MO[i] = self.SV[i] - self.RS[i]
    
    def reconstruct_SV(self):
        """
        Reconstruct the sensitive values using both V_TTR and V_MO.
        """
        reconstructed_SV = (self.V_TTR + self.V_MO) / 2  # Average to cancel out RS
        return reconstructed_SV

    def calculate_sums(self):
        """
        Calculate the sum of V_TTR and V_MO for all job seekers.
        """
        sum_ttr = np.sum(self.V_TTR)
        sum_mo = np.sum(self.V_MO)
        return sum_ttr, sum_mo
    
    def calculate_diversity(self):
        """
        Calculate diversity (DIV) as (sum_ttr + sum_mo) / num_seekers.
        """
        sum_ttr, sum_mo = self.calculate_sums()
        diversity = (sum_ttr + sum_mo) / self.num_seekers
        return diversity

    def calculate_expectation(self, weights):
        """
        Calculate expectation (EXP), weighted by some weights (e.g., importance of categories),
        based on the sum of V_TTR and V_MO.
        """
        sum_ttr, sum_mo = self.calculate_sums()
        expectation = 0
        for k in range(self.num_categories):
            expectation += (sum_ttr + sum_mo) * weights[k]
        return expectation / self.num_seekers

# Example usage:

num_seekers = 3
num_categories = 3

# One-hot encoded sensitive values (SV) for job seekers (A, B, C)
SV = np.array([
    [1, 0, 0],  # Job Seeker A
    [0, 1, 0],  # Job Seeker B
    [0, 0, 1]   # Job Seeker C
])

# Initialize fairness computation class
fairness = FairnessComputation(num_seekers, num_categories)

# Data donation (input sensitive values)
fairness.data_donation(SV)

# Generate V_TTR and V_MO values using random secrets
fairness.generate_V_values()

# Reconstruct the original sensitive values from TTP data
reconstructed_SV = fairness.reconstruct_SV()
print("Reconstructed Sensitive Values (SV):")
print(reconstructed_SV)

# Calculate diversity using sum-based retrieval
diversity = fairness.calculate_diversity()
print(f"Diversity: {diversity}")

# Define some weights for expectation calculation
weights = np.array([1.0, 2.0, 3.0])  # Importance of categories

# Calculate expectation using the weights
expectation = fairness.calculate_expectation(weights)
print(f"Expectation (EXP): {expectation}")
