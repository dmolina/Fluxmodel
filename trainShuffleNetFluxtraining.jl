using DataLoaders: DataLoader
using MLDataPattern: splitobs
using Flux
using FluxTraining
using MLDatasets
using MLDatasets: CIFAR10
using MLDatasets: MNIST
using Flux: onehotbatch
using Optimisers

include("ShuffleNet.jl")

function main()
    
    # data
    xs, ys = CIFAR10()[:]
    ys = onehotbatch(ys, 0:9)

    model = get_shufflenet(8, 2; in_channels=3, num_classes=10)

    # training and validation sets
    traindata, valdata = splitobs((xs, ys))

    # iterators
    trainiter, validiter = DataLoader(traindata, 128, buffered=false), DataLoader(valdata, 256, buffered=false);

    lossfn = Flux.Losses.logitcrossentropy
    #optimizer = Flux.ADAM();
    optimizer = Optimisers.Adam()

    learner = Learner(model, lossfn; callbacks=[ToGPU(), Metrics(accuracy)], optimizer)

    FluxTraining.fit!(learner, 15, (trainiter, validiter))

end

main()
