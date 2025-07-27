using Flux, MLDatasets, Random, Statistics
using Flux: onehotbatch, onecold, flatten, crossentropy, trainable
using CairoMakie
using FileIO

# -------------------- DEVICE SETUP --------------------
const USE_GPU = false  # Set to true if using CUDA
DEV(x) = USE_GPU ? gpu(x) : x

# -------------------- UTILITIES --------------------
BATCH = 128
Random.seed!(45)

onehot(y) = onehotbatch(y, 0:9)

function get_loaders()
    train_x, train_y = CIFAR10(split = :train)[:]
    test_x,  test_y  = CIFAR10(split = :test)[:]

    train_x = Float32.(train_x)
    test_x  = Float32.(test_x)
    train_y = onehot(train_y)
    test_y  = onehot(test_y)

    train_dl = Flux.DataLoader((train_x, train_y); batchsize=BATCH, shuffle=true)
    test_dl  = Flux.DataLoader((test_x,  test_y);  batchsize=BATCH)
    return train_dl, test_dl
end

function compute_accuracy(model, data)
    total, correct = 0, 0
    for (x, y) in data
        preds = onecold(model(DEV(x)))
        labels = onecold(y)
        correct += sum(preds .== labels)
        total += length(labels)
    end
    return correct / total
end

# -------------------- MODELS --------------------
function build_lenet5()
    Chain(
        Conv((5,5), 3=>6, relu),
        MaxPool((2,2)),
        Conv((5,5), 6=>16, relu),
        MaxPool((2,2)),
        flatten,
        Dense(16*5*5, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
        softmax
    ) |> DEV
end

function build_lenet_k(k::Int)
    conv_part = Chain(
        Conv((k,k), 3=>6, relu),
        MaxPool((2,2)),
        Conv((k,k), 6=>16, relu),
        MaxPool((2,2)),
    )
    dummy = conv_part(rand(Float32, 32, 32, 3, 1))
    fc_in = prod(size(dummy)[1:3])
    return Chain(
        conv_part,
        flatten,
        Dense(fc_in, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
        softmax
    ) |> DEV
end

# -------------------- TRAINING FUNCTION (Flux 0.14+) --------------------
function train!(model, train_dl, test_dl; epochs=10, learning_rate=1e-3)
    opt = Flux.Adam(learning_rate)
    ps = Flux.trainable(model)
    opt_state = Flux.setup(opt, ps)

    for epoch in 1:epochs
        for (x, y) in train_dl
            x, y = DEV(x), DEV(y)

            # Compute gradients
            loss, back = Flux.withgradient(ps) do
                y_pred = model(x)
                return Flux.crossentropy(y_pred, y)
            end

            # Apply gradients to update weights
            Flux.update!(opt_state, ps, back)
        end

        acc = compute_accuracy(model, test_dl)
        println("Epoch $epoch | Test Accuracy: $(round(acc * 100; digits=2))%")
    end
end



# -------------------- Q1: LeNet5 Training --------------------
train_dl, test_dl = get_loaders()
model_q1 = build_lenet5()
train!(model_q1, train_dl, test_dl; epochs=10)

# -------------------- Q2: Diversity vs Repetition --------------------
sample_counts = [10_000, 20_000, 30_000]
epochs_map = Dict(10_000 => 6, 20_000 => 3, 30_000 => 2)
accs_q2 = Float64[]

full_x, full_y = CIFAR10(split = :train)[:]
full_x = Float32.(full_x)

for N in sample_counts
    e = epochs_map[N]
    idx = randperm(50_000)[1:N]
    x_sub = full_x[:, :, :, idx]
    y_sub = onehot(full_y[idx])

    dl_sub = Flux.DataLoader((x_sub, y_sub); batchsize=BATCH, shuffle=true)
    model = build_lenet5()
    println("\n>>> Training on $N samples for $e epoch(s)...")
    train!(model, dl_sub, test_dl; epochs=e)
    acc = compute_accuracy(model, test_dl)
    println("Final Test Accuracy: $(round(acc * 100; digits=2)) %")
    push!(accs_q2, acc)
end

fig2 = Figure(resolution=(450, 300))
ax2 = Axis(fig2[1,1], xlabel="# unique samples", ylabel="test accuracy [%]",
           title="Q2: Diversity vs Repetition")
lines!(ax2, sample_counts, accs_q2 .* 100)
scatter!(ax2, sample_counts, accs_q2 .* 100, markersize=10)
save("q2_diversity_vs_repetition.png", fig2)

# -------------------- Q3: Kernel Size Comparison --------------------
kernel_sizes = [3, 5, 7]
accs_q3 = Float64[]

for k in kernel_sizes
    println("\n>>> Training LeNet with $k×$k kernels...")
    model = build_lenet_k(k)
    train!(model, train_dl, test_dl; epochs=4)
    acc = compute_accuracy(model, test_dl)
    println("Final Test Accuracy: $(round(acc * 100; digits=2)) %")
    push!(accs_q3, acc)
end

fig3 = Figure(resolution=(450, 300))
ax3 = Axis(fig3[1,1], xlabel="kernel size", ylabel="test accuracy [%]",
           title="Q3: Filter Size Impact")
lines!(ax3, kernel_sizes, accs_q3 .* 100)
scatter!(ax3, kernel_sizes, accs_q3 .* 100, markersize=10)
save("q3_kernel_vs_accuracy.png", fig3)

# -------------------- Q4: Feature Visualization from LeNet3 --------------------
model_q4 = build_lenet_k(3)
train!(model_q4, train_dl, test_dl; epochs=5)

test_x, _ = CIFAR10(split = :test)[:]
test_x = Float32.(test_x)

for i in [1, 10, 20]
    img = reshape(test_x[:, :, :, i], 32,32,3,1) |> DEV
    conv1 = model_q4.layers[1](img)
    pool1 = model_q4.layers[2](conv1)
    conv2 = model_q4.layers[3](pool1)

    fig = Figure(resolution=(600, 200), fontsize=14)
    Label(fig[0,1], "Image $i - Conv1")
    Label(fig[0,2], "Image $i - Conv2")
    heatmap(fig[1,1], cpu(conv1[:,:,1,1]))
    heatmap(fig[1,2], cpu(conv2[:,:,1,1]))
    save("q4_features_img$i.png", fig)
end
