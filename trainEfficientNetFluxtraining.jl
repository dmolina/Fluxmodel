using DataLoaders: DataLoader
using MLDataPattern: splitobs
using Flux
using FluxTraining
using MLDatasets
using MLDatasets: CIFAR10
using MLDatasets: MNIST
using Flux: onehotbatch
using Optimisers

include("EfficientNet.jl")

function main()

    #data
    xs, ys = CIFAR10()[:]
    ys = onehotbatch(ys, 0:9)

    #model EfficientNet Lite B0
    model = EfficientNet(3, 1, 1, 10, 0)

    # training and validation sets
    traindata, valdata = splitobs((xs, ys))

    #  iterators
    tainiter, validiter = DataLoader(traindata, 128, buffered=false), DataLoader(valdata, 256, buffered=false);

    lossfn = Flux.Losses.logitcrossentropy
    #optimizer = Flux.ADAM();
    optimizer = Optimisers.Adam()

    learner = Learner(model, lossfn; callbacks=[ToGPU(), Metrics(accuracy)], optimizer)

    FluxTraining.fit!(learner, 15, (trainiter, validiter))

end

main()


