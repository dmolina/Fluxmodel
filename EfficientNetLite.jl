using Flux


function round_filters(filters::Number, multiplier::Number; divisor=8, min_width=nothing)
  """Calculate and round number of filters based on width multiplier."""
  filters *= multiplier
  min_width = min_width==nothing ? divisor : min_width
  new_filters = max(min_width, trunc(filters + divisor/2) ÷ divisor*divisor)
  new_filters = new_filters < 0.9*filters ? new_filters+divisor : new_filters
  return trunc(Int, new_filters)
end

function round_repeats(repeats, multiplier)
  """Round number of filters based on depth multiplier."""
  return ceil(multiplier*repeats)
end

"""
MBConvBlock(inplanes::Integer, squeeze_planes::Integer, expand1x1_planes::Integer,
         expand3x3_planes::Integer)

Create a MBConvBlock module
([reference](https://arxiv.org/abs/1801.04381v4)).

# Arguments

  - `io_channels`: tuple of in channels and out channels
"""

function MBConvBlock(k::Tuple{Vararg{Integer, N}},io_channels::Pair{<:Integer, <:Integer}, s::Integer, exp_ratio::Number, se_ratio; se = false, drop_connect_rate=nothing) where N

  moment = 0.01
  epsilon = 1e-3
  exp = io_channels[1]*exp_ratio
  exp = trunc(Int, exp)
  id_skip = true

  m = Chain(Conv((1,1), io_channels[1] => exp, relu; bias=false), #Expansion
            BatchNorm(exp; eps=epsilon, momentum=moment),
            DepthwiseConv(k, exp => exp, relu; stride=s, bias=false, pad=SamePad()), #Depthwise
            BatchNorm(exp; eps=epsilon, momentum=moment)
            )

  #squeeze
  if se
    squeezed = (io_channels[1]*se_ratio) > 1 ? (io_channels[1]*se_ratio) : 1
    squeezed = trunc(Int, squeezed)
    Chain(m,
          AdaptiveMeanPool((1,1),),
          Conv((1,1), exp => squeezed, relu),
          Conv((1,1), squeezed => exp, sigmoid)
          )
    
  end
  #output phase
  m = Chain(m,
            Conv((1,1), exp => io_channels[2], relu; bias=false),
            BatchNorm(io_channels[2]; eps=epsilon, momentum=moment))
"""
Dropout de conexeciones queda por arreglar
  if id_skip && s == 1 && io_channels[1] == io_channels[2]
    if drop_connect_rate != nothing

    end
  end
"""
  if io_channels[1] == io_channels[2]
    return SkipConnection(m, +)
  end

  return m
end


function EfficientNet(widthi_multiplier, depth_multiplier, num_classes, dropout_rate; drop_connect_rate=nothing)
  moment = 0.01
  epsilon = 1e-3
  mb_block_settings =
  #repeat|kernal_size|stride|expand|input|output|se_ratio
      [1 3 1 1 32 16 0.25;
      2 3 2 6 16 24 0.25;
      2 5 2 6 24 40 0.25;
      3 3 2 6 40 80 0.25;
      3 5 1 6 80 112 0.25;
      4 5 2 6 112 192 0.25;
      1 3 1 6 192 320 0.25]

    #Stem
    out_channels = 32
    stem = Chain(
      Conv((3,3), 3 => out_channels, relu; stride=2, pad=1, bias=false),
      BatchNorm(out_channels; eps=epsilon, momentum=moment)
    )
    model = Chain(stem)

    #Blocks
    blocks = []
    for config in eachrow(mb_block_settings)
      num_repeat, kernal, stride, expand_ratio, inputs, outputs, se_ratio = config
      inputs = inputs == 32 ? inputs : round_filters(inputs, widthi_multiplier)
      outputs = round_filters(outputs, widthi_multiplier)
      num_repeat = (inputs==32 || inputs==192) ? num_repeat : round_repeats(num_repeat, depth_multiplier)

      inputs = trunc(Int, inputs)
      outputs = trunc(Int, outputs)
      kernal = trunc(Int, kernal)
      stride = trunc(Int, stride)

      stage = []
      model = Chain(model, MBConvBlock((kernal,kernal), inputs => outputs, stride, expand_ratio, se_ratio; se=false))
      if num_repeat > 1
        inputs = outputs
        stride = 1
      end
      for i in 1:(num_repeat-1)
        model = Chain(model, MBConvBlock((kernal,kernal),inputs => outputs, stride, expand_ratio, se_ratio; se=false))
      end
    end

    #Head
    in_channels = round_filters(mb_block_settings[7,6], widthi_multiplier)
    out_channels::Integer = 1280
    head = Chain(
      Conv((1,1), in_channels => out_channels, relu; stride=1, pad=0, bias=false),
      BatchNorm(out_channels; eps=epsilon, momentum=moment)
    )

    avgpool = AdaptiveMeanPool((1, 1))

    model = Chain(model, head, avgpool)

    if dropout_rate > 0
        dropout = Dropout(dropout_rate)
        model = Chain(model, dropout)
    end

    model = Parallel(x -> reshape(x, (size(x,3),size(x,4))), model)

    fc = Dense(out_channels => num_classes)


    model = Chain(model, fc)

    return model
end
