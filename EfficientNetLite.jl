using Flux
include("MBConvBlock.jl")

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


function EfficientNet(input_channels::Int, widthi_multiplier, depth_multiplier, num_classes, dropout_rate)
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

    blocks = []
    #Stem
    out_channels = 32
    append!(blocks,[Conv((3,3), input_channels => out_channels; stride=2, pad=1, bias=false),
    BatchNorm(out_channels; eps=epsilon, momentum=moment),
    NNlib.relu])

    #Blocks
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
      push!(stage, MBConvBlock((kernal,kernal), inputs => outputs, stride, expand_ratio))
      if num_repeat > 1
        inputs = outputs
        stride = 1
      end
      for i in 1:(num_repeat-1)
        append!(stage, [MBConvBlock((kernal,kernal),inputs => outputs, stride, expand_ratio)])
      end
      append!(blocks, stage)
    end

    model = Chain(blocks...)
    #Head
    in_channels = round_filters(mb_block_settings[7,6], widthi_multiplier)
    out_channels::Integer = 1280
    head = Chain(
      Conv((1,1), in_channels => out_channels; stride=1, pad=0, bias=false),
      BatchNorm(out_channels; eps=epsilon, momentum=moment),
      NNlib.relu)

    avgpool = AdaptiveMeanPool((1, 1))

    model = Chain(model, head, avgpool)

    if dropout_rate > 0
        dropout = Dropout(dropout_rate)
        model = Chain(model, dropout)
    end

    model = Chain(model, Flux.flatten, Dense(out_channels => num_classes))

    return model
end
