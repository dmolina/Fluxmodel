using MLDatasets, Flux, JLD2, CUDA

"""
folder = "runs"
isdir(folder) || mkdir(folder)
filename = joinpath(folder, "model.jld2")
"""

train_data = MLDatasets.MNIST() 
test_data = MLDatasets.MNIST(split=:test)


function loader(data::MNIST=train_data; batchsize::Int=64)
    x4dim = reshape(data.features, 28,28,1,:)
    yhot = Flux.onehotbatch(data.targets, 0:9) 
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true) |> gpu
end
x1, y1 = first(loader())

function MBConvBlock(k::Tuple{Vararg{Integer, N}},io_channels::Pair{<:Integer, <:Integer}, s::Integer, exp_ratio::Number) where N

    moment = 0.01
    epsilon = 1e-3
    exp = io_channels[1]*exp_ratio
    exp = trunc(Int, exp)
  
    m = Chain(Conv((1,1), io_channels[1] => exp; bias=false), #Expansion
              BatchNorm(exp; eps=epsilon, momentum=moment),
              NNlib.relu,
              DepthwiseConv(k, exp => exp; stride=s, bias=false, pad=SamePad()), #Depthwise
              BatchNorm(exp; eps=epsilon, momentum=moment),
              NNlib.relu,

              Conv((1,1), exp => io_channels[2]; bias=false),#output
              BatchNorm(io_channels[2]; eps=epsilon, momentum=moment),
              NNlib.relu)

    if io_channels[1] == io_channels[2]
      return SkipConnection(m, +)
    end
    return m
  end

  model = Chain(Conv((3,3), 1 => 32; stride=2, pad=1, bias=false), #Stem
                  BatchNorm(32; eps=1e-3, momentum=0.01),
                  NNlib.relu,
                  
                  MBConvBlock((3,3), 32 => 16, 1, 1), #Blocks

                  MBConvBlock((3,3), 16 => 24, 2, 6),
                  MBConvBlock((3,3), 24 => 24, 1, 6),
      
                  MBConvBlock((5,5), 24 => 40, 2, 6),
                  MBConvBlock((5,5), 40 => 40, 1, 6),

                  MBConvBlock((3,3), 40 => 80, 2, 6),
                  MBConvBlock((3,3), 80 => 80, 1, 6),
                  MBConvBlock((3,3), 80 => 80, 1, 6),

                  MBConvBlock((5,5), 80 => 112, 1, 6),
                  MBConvBlock((5,5), 112 => 112, 1, 6),
                  MBConvBlock((5,5), 112 => 112, 1, 6),

                  MBConvBlock((5,5), 112 => 192, 2, 6),
                  MBConvBlock((5,5), 192 => 192, 1, 6),
                  MBConvBlock((5,5), 192 => 192, 1, 6),
                  MBConvBlock((5,5), 192 => 192, 1, 6),

                  MBConvBlock((3,3), 192 => 320, 1, 6),

                  Conv((1,1), 320 => 1280; stride=1, pad=0, bias=false), #Head
                  BatchNorm(1280; eps=1e-3, momentum=0.01),
                  NNlib.relu,

                  AdaptiveMeanPool((1, 1)),

                  Flux.flatten,

                  Dense(1280 => 10)       

    ) |> gpu


"""
include("ShuffleNet.jl")

model = get_shufflenet(2, 1; in_channels=1, in_size=(28,28), num_classes=10) |> gpu
"""

y1hat = model(x1)

sum(softmax(y1hat); dims=1)

@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

using Statistics: mean  

function loss_and_accuracy(model, data::MNIST=test_data)
    (x,y) = only(loader(data; batchsize=length(data)))
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split=data.split)  
end

@show loss_and_accuracy(model); 

settings = (;
    eta = 3e-4,
    lambda = 1e-2,  
    batchsize = 128,
    epochs = 10,
)
train_log = []


opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
opt_state = Flux.setup(opt_rule, model);

for epoch in 1:settings.epochs
    @time for (x,y) in loader(batchsize=settings.batchsize)
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
        Flux.update!(opt_state, model, grads[1])
    end

    if epoch % 2 == 1
        loss, acc, _ = loss_and_accuracy(model)
        test_loss, test_acc, _ = loss_and_accuracy(model, test_data)
        @info "logging:" epoch acc test_acc
        nt = (; epoch, loss, acc, test_loss, test_acc)
        push!(train_log, nt)
    end
    #if epoch % 5 == 0
    #    JLD2.jldsave(filename; model_state = Flux.state(model) |> cpu)
    #    println("saved to ", filename, " after ", epoch, " epochs")
    #end
end

@show train_log;
