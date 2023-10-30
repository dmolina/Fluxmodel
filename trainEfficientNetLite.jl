using FastAI, FastVision, FastTabular, FastMakie, Metalhead, Optimisers
import CairoMakie; CairoMakie.activate!(type="png")
include("EfficientNet.jl")

data, blocks = load(datarecipes()["cifar10"])

task = ImageClassificationSingle(blocks, size=(128, 128))

backbone = EfficientNet(3, 1.4, 1.8, 10, 0.3)

learner = tasklearner(task, data; callbacks=[ToGPU(), Metrics(accuracy)], optimizer=Optimisers.Adam())

fitonecycle!(learner, 15, 0.004)

savetaskmodel("./EfficientNetLite4-1-cifar10.jld2", task, learner.model)
