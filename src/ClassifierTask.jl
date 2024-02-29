using Flux
using CSV
using DataFrames
using Plots
using Statistics
using Random

Random.seed!(123)

# Define the neural network architecture
model = Chain(
    Dense(3, 32, σ),
    Dense(32, 2),
    softmax
)

# Loss function
loss(x, y) = Flux.crossentropy(model(x), y)

# Optimizer
opt = ADAM()

# Read CSV file
data = CSV.File("./data/dataset.csv") |> DataFrame

# Preprocessing
X = Matrix(data[:, 1:3])
X = (X .- mean(X, dims=1)) ./ std(X, dims=1)  # Standardize each column
X = transpose(X)  # Transpose X to match expected dimensions (3, num_samples)
y = Flux.onehotbatch(data[:, :class], ["s", "b"])

# Split data into training and testing sets (80-20 split)
split_index = Int(0.8 * size(X, 2))
X_train, X_test = X[:, 1:split_index], X[:, split_index+1:end]
y_train, y_test = y[:, 1:split_index], y[:, split_index+1:end]

# Train the model
epochs = 10
train_loss = zeros(epochs)
accuracy = zeros(epochs)
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), [(X_train, y_train)], opt)
    train_loss[epoch] = loss(X_train, y_train)
    accuracy[epoch] = sum(Flux.onecold(model(X_test)) .== Flux.onecold(y_test)) / size(y_test, 2)
    @info "Epoch $epoch, Loss: $(train_loss[epoch]), Accuracy: $(accuracy[epoch])"
end

# Plotting the loss during training
lossPlot = plot(1:epochs, train_loss, xlabel="Epoch", ylabel="Loss", label="Training Loss", legend=:bottomright)

savefig(lossPlot, "training_loss_plot.png")

# Evaluate accuracy on test set
println("Final Accuracy on Test Set: ", accuracy[end])
