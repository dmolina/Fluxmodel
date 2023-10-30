using FastAI, FastVision, FastTabular, FastMakie, Metalhead, Optimisers
import CairoMakie; CairoMakie.activate!(type="png")
include("ShuffleNet.jl")

data, blocks = load(datarecipes()["cifar10"])

task = ImageClassificationSingle(blocks, size=(128, 128))

backbone = get_shufflenet(1, 1; in_channels=3, num_classes=10)

learner = tasklearner(task, data; callbacks=[ToGPU(), Metrics(accuracy)], optimizer=Optimisers.Adam())

fitonecycle!(learner, 10, 0.004)

savetaskmodel("./ShuffleNet1-1-cifar10.jld2", task, learner.model)
