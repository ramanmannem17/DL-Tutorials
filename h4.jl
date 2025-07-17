# HW4: FashionMNIST Classification using Lux.jl

using Lux, Optimisers, MLDatasets, Flux, Zygote, Statistics, Random, Plots, DataFrames, PrettyTables

# ---------------------- Data Prep ----------------------
Random.seed!(42)
train_imgs, train_lbls = FashionMNIST.traindata(); test_imgs, test_lbls = FashionMNIST.testdata()

function normalize(images)
    Float32.(reshape(images, :, size(images, 3))) ./ 255
end

X_train = normalize(train_imgs)
Y_train = Flux.onehotbatch(train_lbls .+ 1, 1:10)
X_test = normalize(test_imgs)
Y_test = Flux.onehotbatch(test_lbls .+ 1, 1:10)

function accuracy(model_fn, X, Y)
    preds = model_fn(X)
    mean(Flux.onecold(preds) .== Flux.onecold(Y))
end

# -------------------- Q1: Vary Hidden Layer Sizes --------------------
epochs = 10
batch_sz = 128
hidden_dims = [10, 20, 40, 50, 100, 300]
acc_scores = Float32[]

function train_single_layer(hidden_dim)
    model = Chain(Dense(28^2, hidden_dim, relu), Dense(hidden_dim, 10))
    ps, st = Lux.setup(RNG(), model)
    opt_state = Optimisers.setup(Optimisers.Adam(0.01), ps)

    for _ in 1:epochs
        for i in 1:batch_sz:size(X_train, 2)
            idx = i:min(i + batch_sz - 1, size(X_train, 2))
            x, y = X_train[:, idx], Y_train[:, idx]

            loss_fn(p) = Flux.logitcrossentropy(first(model(x, p, st)), y)
            grads = Zygote.gradient(loss_fn, ps)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
        end
    end

    return accuracy(x -> first(model(x, ps, st)), X_test, Y_test)
end

for h in hidden_dims
    push!(acc_scores, train_single_layer(h))
end

plot(hidden_dims, acc_scores, xlabel="Hidden Size", ylabel="Accuracy", label="", marker=:circle, lw=2)

# ------------------ Q2: Random Init Impact ------------------
fixed_hidden = 30
num_trials = 10
seeds = rand(1:10^6, num_trials)
acc_variants = Float32[]

for s in seeds
    Random.seed!(s)
    push!(acc_variants, train_single_layer(fixed_hidden))
end

mean_acc = mean(acc_variants)
std_acc = std(acc_variants)

println("Mean Accuracy = $(round(mean_acc*100, digits=2))%, Std = $(round(std_acc*100, digits=2))%")
scatter(1:num_trials, acc_variants, xlabel="Trial", ylabel="Accuracy", title="Random Init Impact", label="", marker=:diamond)
hline!([mean_acc], linestyle=:dash, label="Mean")

# ------------------ Q3: Learning Rate Decay ------------------
epochs_q3 = 25
bs_q3 = 32

function train_decay(hidden_dim; η=0.01, decay=0.9)
    model = Chain(Dense(28^2, hidden_dim, relu), Dense(hidden_dim, 10))
    ps, st = Lux.setup(RNG(), model)

    for ep in 1:epochs_q3
        η_t = η * decay^(ep - 1)
        opt_state = Optimisers.setup(Optimisers.Adam(η_t), ps)

        for i in 1:bs_q3:size(X_train, 2)
            idx = i:min(i + bs_q3 - 1, size(X_train, 2))
            x, y = X_train[:, idx], Y_train[:, idx]
            loss_fn(p) = Flux.logitcrossentropy(first(model(x, p, st)), y)
            grads = Zygote.gradient(loss_fn, ps)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
        end
    end

    return accuracy(x -> first(model(x, ps, st)), X_test, Y_test)
end

acc_q3 = train_decay(50)
println("Q3 Accuracy (bs=32, decay): $(round(acc_q3*100, digits=2))%")

# ------------------ Q4: Grid Search ------------------
bsizes = [16, 32, 64, 128]
rates = [0.001, 0.005, 0.01]
decays = [0.9, 0.95, 1.0]

results = []

for b in bsizes, lr in rates, d in decays
    acc = train_decay(50; η=lr, decay=d)
    push!(results, (b, lr, d, acc))
    println("Grid: bs=$b, lr=$lr, decay=$d -> acc=$(round(acc*100, digits=2))%")
end

df = DataFrame(results, [:BatchSize, :LR, :Decay, :Accuracy])
pretty_table(df)

# ------------------ Q5: Best Params ------------------
best = sort(df, :Accuracy, rev=true)[1, :]
acc_best = train_decay(50; η=best.LR, decay=best.Decay)

println("Best Grid Config: bs=$(best.BatchSize), lr=$(best.LR), decay=$(best.Decay)")
println("Final Accuracy using best config: $(round(acc_best*100, digits=2))%")

if acc_best > acc_q3
    println(" Improved over Q3 setup")
else
    println(" No improvement")
end
