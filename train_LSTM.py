
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_pickle("./dataset/training_data_landfrac_1.pkl")

# List of columns for time-series data
time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
static_columns = [col for col in df.columns if col not in time_series_columns and not col.startswith('Y_')]
target_columns = [col for col in df.columns if col.startswith('Y_')]

# Ensure each time-series column is a consistent numpy array of floats with length 29200
for col in time_series_columns:
    df[col] = df[col].apply(
        lambda x: np.pad(x[:29200], (0, max(0, 29200 - len(x))), 'constant') if isinstance(x, list) else np.zeros(29200, dtype=np.float32)
    )

# Shuffle the dataframe before splitting into train and test sets
df = shuffle(df, random_state=42).reset_index(drop=True)

# Convert the time-series data into a 3D numpy array (samples, time_steps, features)
time_series_data = np.stack([np.column_stack(df[col]) for col in time_series_columns], axis=-1)

# Normalize the time-series data across each feature
scaler_time_series = StandardScaler()
time_series_data = time_series_data.reshape(-1, len(time_series_columns))  # Flatten for normalization
time_series_data = scaler_time_series.fit_transform(time_series_data)
time_series_data = time_series_data.reshape(-1, 29200, len(time_series_columns))  # Reshape back

# Normalize the static and target features
scaler_static = StandardScaler()
static_data = scaler_static.fit_transform(df[static_columns].values)

scaler_target = StandardScaler()
target_data = scaler_target.fit_transform(df[target_columns].values)

# Convert data to PyTorch tensors
time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
static_data = torch.tensor(static_data, dtype=torch.float32)
target_data = torch.tensor(target_data, dtype=torch.float32)

# Split data into train and test sets
train_size = int(0.8 * len(df))
test_size = len(df) - train_size

train_time_series = time_series_data[:train_size]
train_static = static_data[:train_size]
train_target = target_data[:train_size]

test_time_series = time_series_data[train_size:]
test_static = static_data[train_size:]
test_target = target_data[train_size:]

# Move training tensors to GPU
train_time_series = train_time_series.to(device)
train_static = train_static.to(device)
train_target = train_target.to(device)

# Define the model
class CombinedModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, num_static_features, fc_hidden_size, output_size):
        super(CombinedModel, self).__init__()
        
        # LSTM branch for time-series data
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True)
        
        # Fully connected layer for static features
        self.fc_static = nn.Linear(num_static_features, fc_hidden_size)
        
        # Fully connected layers for combined output
        self.fc1 = nn.Linear(lstm_hidden_size + fc_hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)  # Output layer
    
    def forward(self, time_series_data, static_data):
        # LSTM branch
        lstm_out, _ = self.lstm(time_series_data)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state
        
        # Static feature branch
        static_out = torch.relu(self.fc_static(static_data))
        
        # Concatenate both branches
        combined = torch.cat((lstm_out, static_out), dim=1)
        
        # Pass through final fully connected layers
        x = torch.relu(self.fc1(combined))
        output = self.fc2(x)
        
        return output

# Model parameters
lstm_input_size = len(time_series_columns)
lstm_hidden_size = 64
num_static_features = static_data.shape[1]
fc_hidden_size = 32
output_size = len(target_columns)

# Initialize the model and move it to the GPU
model = CombinedModel(lstm_input_size, lstm_hidden_size, num_static_features, fc_hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 16  # Reduced batch size to save memory
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0  # Accumulate training loss
    for i in range(0, train_size, batch_size):
        # Batch data
        end_idx = min(i + batch_size, train_size)
        time_series_batch = train_time_series[i:end_idx]
        static_batch = train_static[i:end_idx]
        target_batch = train_target[i:end_idx]

        # Forward pass
        optimizer.zero_grad()
        output = model(time_series_batch, static_batch)
        loss = criterion(output, target_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()

    # Calculate average training loss for this epoch
    avg_train_loss = running_train_loss / (train_size / batch_size)
    train_losses.append(avg_train_loss)

    # Validation loss with batch processing
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for j in range(0, test_size, batch_size):
            end_idx = min(j + batch_size, test_size)
            time_series_batch = test_time_series[j:end_idx].to(device)
            static_batch = test_static[j:end_idx].to(device)
            target_batch = test_target[j:end_idx].to(device)
            
            val_output = model(time_series_batch, static_batch)
            val_loss += criterion(val_output, target_batch).item()
        
        val_loss /= (test_size / batch_size)
        val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save training and validation losses to CSV
losses_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_losses,
    'Validation Loss': val_losses
})
losses_df.to_csv("./results/training_validation_losses2.csv", index=False)
print("Training and validation losses saved as './results/training_validation_losses2.csv'")

# Evaluation on the test set
model.eval()
predictions = []
with torch.no_grad():
    for i in range(0, test_size, batch_size):
        end_idx = min(i + batch_size, test_size)
        time_series_batch = test_time_series[i:end_idx].to(device)
        static_batch = test_static[i:end_idx].to(device)
        predictions.append(model(time_series_batch, static_batch).cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
# test_loss = mean_squared_error(target_data[train_size:].cpu(), predictions, squared=False)
test_loss = mean_squared_error(target_data[train_size:].cpu(), predictions)
print(f"Test Loss: {test_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "./models/combined_model2.pth")
print("Model saved as './models/combined_model2.pth'")

# Inverse transform predictions and ground truth to original scale
predictions_np = scaler_target.inverse_transform(predictions)
ground_truth_np = scaler_target.inverse_transform(test_target.cpu().numpy())

# Convert predictions and ground truth to DataFrames and save
predictions_df = pd.DataFrame(predictions_np, columns=target_columns)
ground_truth_df = pd.DataFrame(ground_truth_np, columns=target_columns)

# Save as CSV files
predictions_df.to_csv("./results/predictions_original_scale2.csv", index=False)
ground_truth_df.to_csv("./results/ground_truth_original_scale2.csv", index=False)

print("Predictions saved as './results/predictions_original_scale2.csv'")
print("Ground truth saved as './results/ground_truth_original_scale2.csv'")


# Extract coordinates (latitude and longitude) for corresponding rows
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    coordinates_df = df[['Latitude', 'Longitude']].iloc[train_size:].reset_index(drop=True)
    # Save the coordinates for the test dataset
    coordinates_df.to_csv("coordinates2.csv", index=False)
    print("Coordinates saved as 'coordinates2.csv'")
else:
    print("Latitude and Longitude columns are not present in the DataFrame.")

# Create the figure and plot the data
plt.figure(figsize=(10, 6))

# Increase line thickness by setting linewidth
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', linewidth=2)

# Set labels with larger font sizes
plt.xlabel('Epoch', fontsize=18, weight='bold')
plt.ylabel('Mean Squared Error (MSE) Loss', fontsize=18, weight='bold')

# Set title with a larger font size and bold text
plt.title('Training and Validation Loss vs. Epoch', fontsize=20, weight='bold')

# Enhance legend with larger font size
plt.legend(fontsize=18, loc='upper right', frameon=True)

# Increase tick font size for better readability
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Add grid for easier reading of data points
plt.grid(True, linestyle='--', linewidth=0.5)

# Adjust layout and save the figure
plt.tight_layout()
output_file = "./results/training_validation_losses_plot.png"
plt.savefig(output_file, dpi=300)
print(f"Figure saved as {output_file}")

# Close the figure to free memory
plt.close()