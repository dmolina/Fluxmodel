using Flux

function ChannelShuffle(x,g)
    println(x," ",g)
    width, height, channels, batch = size(x)
    channels_per_group = channels÷g
    if (channels % g) == 0
        x = reshape(x, (width, height, g, channels_per_group, batch))
        x = permutedims(x(1,2,4,3,5))
        x = reshape(x, (width, height, channels, batch))
    end
    return x
end

function ShuffleUnit(in_channels::Integer, out_channels::Integer, grps::Integer, downsample::Bool, ignore_group::Bool)
    mid_channels = out_channels ÷ 4
    grps = ignore_group ? 1 : grps
    strd = downsample ? 2 : 1
    
    if downsample
        out_channels -= in_channels
    end

    m = Chain(Conv((1,1), in_channels => mid_channels, relu; groups=grps),
              BatchNorm(mid_channels),
              ChannelShuffle(mid_channels, grps),
              DepthwiseConv((3,3), 3 => 6, relu; bias=false, stride=strd),
              BatchNorm(mid_channels),
              Conv((1,1), in_channels => mid_channels, relu; groups=grps),
              BatchNorm(mid_channels)
    )
    if downsample
        sm = SkipConnection(m, (mx, x) -> cat(mx, x, dims=3));
        m = Parallel((mx, x) -> cat(mx, x, dims=3),m, MeanPool((3,3); pad=1, stride=1))
    else
        m = SkipConnection(m, +)
    end
    return m
end

function ShuffleInitBlock(in_channels, out_channels)
    m = Chain(Conv((3,3), in_channels => out_channels, relu; stride=2),
              BatchNorm(out_channels),
              MaxPool((3,3); stride=2, pad=1)
    )
    return m
end


function ShuffleNet(channels, init_block_channels, groups; in_channels=3, in_size=(224,224), num_classes=1000)
    init = ShuffleInitBlock(in_channels, init_block_channels)
    in_channels = init_block_channels
    features = []
    for (i, num_channels) in enumerate(channels)
        stage = []
        for (j, out_channels) in enumerate(num_channels)
            downsample = j==1
            ignore_group = i==1 && j==1
            append!(stage, [ShuffleUnit(in_channels, init_block_channels, groups, downsample, ignore_group)])
            in_channels = out_channels
        end
        append!(features, stage)
    end
    final = MeanPool((7,7); stride=1)
    output = Dense(in_channels => num_classes)
    return Chain(init, features, final, output)
end
