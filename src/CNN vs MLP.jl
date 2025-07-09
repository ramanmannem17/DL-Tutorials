using Flux, CSV, DataFrames, Statistics

# ========== CONFIGURATION ==========

folder = "mnist"

# ========== LOAD DATA ==========

println("Loading data...")
train_data = CSV.read(joinpath(folder, "mnist_train.csv"), DataFrame)
test_data = CSV.read(joinpath(folder, "mnist_test.csv"), DataFrame)

# ========== DATA LOADERS ==========

function loader_cnn(data::DataFrame; batchsize=512)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1,2))
    yhot = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x4dim, yhot); batchsize=batchsize, shuffle=true)
end

function loader_mlp(data::DataFrame; batchsize=512)
    x = Matrix{Float32}(select(data, Not(:label)))' ./ 255.0
    y = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x, y); batchsize=batchsize, shuffle=true)
end

# ========== MODELS ==========

lenet = Chain(
    Conv((5, 5), 1 => 6, relu),
    MeanPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MeanPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10)
)

mlp = Chain(
    Flux.flatten,
    Dense(784 => 52, relu),
    Dense(52 => 14, relu),
    Dense(14 => 10)
)

# ========== LOSS & ACCURACY ==========

function loss_and_accuracy(model, data_loader)
    (x, y) = only(data_loader)
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    return loss, acc
end

# ========== TRAINING ==========

function train_flux!(model, train_loader, eval_loader; epochs=10)
    opt = ADAM(0.001)
    state = Flux.setup(opt, model)   # new setup
    start_time = time()

    for epoch in 1:epochs
        for (x, y) in train_loader
            # Compute gradients w.r.t. model directly
            loss, back = Flux.withgradient(model) do m
                Flux.logitcrossentropy(m(x), y)
            end
            # Update parameters
            state, model = Flux.update!(state, model, back[1])
        end

        if epoch % 2 == 1
            train_loss, train_acc = loss_and_accuracy(model, eval_loader)
            println("Epoch $epoch: Train loss = $train_loss, Train acc = $train_acc%")
        end
    end

    return round(time() - start_time, digits=2)
end

# ========== RUN TRAINING ==========

# CNN Training
println("\nTraining CNN (LeNet)...")
train_loader_cnn = loader_cnn(train_data; batchsize=64)
eval_loader_cnn = loader_cnn(train_data; batchsize=size(train_data, 1))
cnn_time = train_flux!(lenet, train_loader_cnn, eval_loader_cnn; epochs=10)
println("CNN training time: $cnn_time seconds")

# MLP Training
println("\nTraining MLP...")
train_loader_mlp = loader_mlp(train_data; batchsize=64)
eval_loader_mlp = loader_mlp(train_data; batchsize=size(train_data, 1))
mlp_time = train_flux!(mlp, train_loader_mlp, eval_loader_mlp; epochs=10)
println("MLP training time: $mlp_time seconds")

# ========== SUMMARY ==========

println("\nSummary:")
if cnn_time < mlp_time
    println("CNN is faster by $(mlp_time - cnn_time) seconds")
else
    println("MLP is faster by $(cnn_time - mlp_time) seconds")
end
