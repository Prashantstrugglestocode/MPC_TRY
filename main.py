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
        self.SV = SV

    def generate_V_values(self):
        for i in range(self.num_seekers):
            self.V_TTR[i] = self.SV[i] + self.RS[i]
            self.V_MO[i] = self.SV[i] - self.RS[i]
    
    def reconstruct_SV(self):
        """
        Reconstruct the sensitive values using both V_TTR and V_MO.
        """
        reconstructed_SV = (self.V_TTR + self.V_MO) / 2  # Average to cancel out RS
        return reconstructed_SV
    
    def calculate_diversity(self, category_k):
        category_data = self.SV[:, category_k]  # Extract category k data
        diversity = np.sum(category_data) / self.num_seekers
        return diversity

    def calculate_expectation(self, weights):
        expectation = 0
        for i in range(self.num_seekers):
            for k in range(self.num_categories):
                if self.SV[i, k] == 1:  # If the SV is in this category
                    expectation += weights[k]
        return expectation

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

# Calculate diversity for category 0
div_category_0 = fairness.calculate_diversity(category_k=0)
print(f"Diversity for category 0: {div_category_0}")

# Define some weights for expectation calculation
weights = np.array([1.0, 2.0, 3.0])  # Importance of categories

# Calculate expectation using the weights
expectation = fairness.calculate_expectation(weights)
print(f"Expectation (EXP): {expectation}")
