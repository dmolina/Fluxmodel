using Flux

function Channelshuffle(x,g)
    width, height, channels, batch = size(x)
    channels_per_group = channels÷g
    if (channels % g) == 0
        x = reshape(x, (width, height, g, channels_per_group, batch))
        x = permutedims(x,(1,2,4,3,5))
        x = reshape(x, (width, height, channels, batch))
    end
    return x
end

struct ChannelShuffle
    chain::Chain
    g::Integer
end
  
function (m::ChannelShuffle)(x)
    return Channelshuffle(m.chain(x),m.g)
end
  
Flux.@functor ChannelShuffle

function ShuffleUnit(in_channels::Integer, out_channels::Integer, grps::Integer, downsample::Bool, ignore_group::Bool)
    mid_channels = out_channels ÷ 4
    grps = ignore_group ? 1 : grps
    strd = downsample ? 2 : 1

    if downsample
        out_channels -= in_channels
    end

    chain = Chain(Conv((1,1), in_channels => mid_channels, relu; groups=grps,pad=SamePad()),
              BatchNorm(mid_channels))
    m = ChannelShuffle(chain, grps)
    m = Chain(m,
              DepthwiseConv((3,3),  mid_channels => mid_channels, relu; bias=false, stride=strd, pad=SamePad()),
              BatchNorm(mid_channels),
              Conv((1,1), mid_channels => out_channels, relu; groups=grps, pad=SamePad()),
              BatchNorm(out_channels)
    )
    
    if downsample
        #sm = SkipConnection(m, (mx, x) -> cat(mx, x, dims=3));
        m = Parallel((mx, x) -> cat(mx, x, dims=3),m, MeanPool((3,3); pad=SamePad(), stride=2))
    else
        m = SkipConnection(m, +)
    end
    return m
end

function ShuffleInitBlock(in_channels::Integer, out_channels::Integer)
    m = Chain(Conv((3,3), in_channels => out_channels, relu; stride=2, pad=SamePad()),
              BatchNorm(out_channels),
              MaxPool((3,3); stride=2, pad=SamePad())
    )
    return m
end


function ShuffleNet(channels, init_block_channels::Integer, groups; in_channels=3, in_size=(224,224), num_classes=1000)
    init = ShuffleInitBlock(in_channels, init_block_channels)
    model = Chain(init)
    in_channels::Integer = init_block_channels
    features = []
    for (i, num_channels) in enumerate(channels)
        stage = []
        for (j, out_channels) in enumerate(num_channels)
            downsample = j==1
            ignore_group = i==1 && j==1
            out_ch::Integer = trunc(out_channels)
            #append!(stage, [ShuffleUnit(in_channels, out_ch, groups, downsample, ignore_group)])
            model = Chain(model, ShuffleUnit(in_channels, out_ch, groups, downsample, ignore_group))
            in_channels = out_ch
        end
        #append!(features, stage)
    end



    model = Chain(model, GlobalMeanPool())
    model = Parallel(x -> reshape(x, (size(x, 3),size(x, 4))), model)
    output = Dense(in_channels => num_classes)
    model = Chain(model, output)
    #return Chain(init, features, final, output)
    return model
end

function get_shufflenet(groups, width_scale; in_channels=3, in_size=(224,224), num_classes=1000)
    init_block_channels = 24
    layers = [4, 8, 4]

    if groups == 1
        channels_per_layers = [144, 288, 576]
    elseif groups == 2
        channels_per_layers = [200, 400, 800]
    elseif groups == 3
        channels_per_layers = [240, 480, 960]
    elseif groups == 4
        channels_per_layers = [272, 544, 1088]
    elseif groups == 8
        channels_per_layers = [384, 768, 1536]
    else
        return error("The number of groups is not supported. Groups = ", groups)
    end

    channels = []
    for i in eachindex(layers)
        char = [channels_per_layers[i]]
        new = repeat(char, layers[i])
        push!(channels, new)
    end

    if width_scale != 1.0
        channels = channels*width_scale

        init_block_channels::Integer = trunc(init_block_channels * width_scale)
    end

    net = ShuffleNet(
        channels,
        init_block_channels,
        groups; 
        in_channels=in_channels, 
        in_size=in_size,
        num_classes=num_classes)

    return net
end
