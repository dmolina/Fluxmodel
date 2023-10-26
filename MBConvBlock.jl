using Flux

function MBConvBlock(k::Tuple{Vararg{Integer, N}},io_channels::Pair{<:Integer, <:Integer}, s::Integer, exp_ratio::Number, se_ratio; se = false, drop_connect_rate=nothing) where N

    moment = 0.01
    epsilon = 1e-3
    exp = io_channels[1]*exp_ratio
    exp = trunc(Int, exp)
    id_skip = true
  
    m = Chain(Conv((1,1), io_channels[1] => exp; bias=false), #Expansion
              BatchNorm(exp; eps=epsilon, momentum=moment),
              NNlib.relu,
              DepthwiseConv(k, exp => exp; stride=s, bias=false, pad=SamePad()), #Depthwise
              BatchNorm(exp; eps=epsilon, momentum=moment),
              NNlib.relu)
  
    #squeeze
    if se
      squeezed = (io_channels[1]*se_ratio) > 1 ? (io_channels[1]*se_ratio) : 1
      squeezed = trunc(Int, squeezed)
      Chain(m,
            AdaptiveMeanPool((1,1),),
            Conv((1,1), exp => squeezed, relu),
            Conv((1,1), squeezed => exp, sigmoid))
      
    end
    #output phase
    m = Chain(m,
              Conv((1,1), exp => io_channels[2]; bias=false),
              BatchNorm(io_channels[2]; eps=epsilon, momentum=moment),
              NNlib.relu)
    
    if io_channels[1] == io_channels[2] && s == 1
      return SkipConnection(m, +)
    end
  
    return m
  end
