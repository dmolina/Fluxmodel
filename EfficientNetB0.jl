using Flux
include("MBConvBlock.jl")


function EfficientNet(num_classes, dropout_rate; drop_connect_rate=nothing)
  moment = 0.01
  epsilon = 1e-3

    #Stem
    out_channels = 32
    stem = Chain(
      Conv((3,3), 3 => out_channels, relu; stride=2, pad=1, bias=false),
      BatchNorm(out_channels; eps=epsilon, momentum=moment)
    )
    model = Chain(stem)

    """
    MBConv1, k3x3 112 × 112 16 1
    MBConv6, k3x3 112 × 112 24 2
    MBConv6, k5x5 56 × 56 40 2
    MBConv6, k3x3 28 × 28 80 3
    MBConv6, k5x5 14 × 14 112 3
    MBConv6, k5x5 14 × 14 192 4
    MBConv6, k3x3 7 × 7 320 1
    """

    #Blocks

    model = Chain(model,MBConvBlock((3,3), 32 => 16, 1, 1, se_ratio; se=false),

                MBConvBlock((3,3), 16 => 16, 2, 6, se_ratio; se=false),
                MBConvBlock((3,3), 16 => 24, 2, 6, se_ratio; se=false),
    
                MBConvBlock((5,5), 24 => 24, 2, 6, se_ratio; se=false),
                MBConvBlock((5,5), 24 => 40, 2, 6, se_ratio; se=false),

                MBConvBlock((3,3), 40 => 40, 2, 6, se_ratio; se=false),
                MBConvBlock((3,3), 40 => 40, 2, 6, se_ratio; se=false),
                MBConvBlock((3,3), 40 => 80, 2, 6, se_ratio; se=false),

                MBConvBlock((5,5), 80 => 80, 1, 6, se_ratio; se=false),
                MBConvBlock((5,5), 80 => 80, 1, 6, se_ratio; se=false),
                MBConvBlock((5,5), 80 => 112, 1, 6, se_ratio; se=false),

                MBConvBlock((5,5), 112 => 112, 2, 6, se_ratio; se=false),
                MBConvBlock((5,5), 112 => 112, 2, 6, se_ratio; se=false),
                MBConvBlock((5,5), 112 => 112, 2, 6, se_ratio; se=false),
                MBConvBlock((5,5), 112 => 192, 2, 6, se_ratio; se=false),

                MBConvBlock((3,3), 192 => 320, 1, 6, se_ratio; se=false)
    )

    #Head
    in_channels = 320
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
