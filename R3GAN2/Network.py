import jax
from flax import nnx
import jax.numpy as jnp
from jax import lax
from .MagnitudePreservingLayers import BiasedPointwiseConvolutionWithModulation, NoisyBiasedPointwiseConvolutionWithModulation, LeakyReLU, BoundedParameter, Linear, GenerativeBasis, DiscriminativeBasis, ClassEmbedder, Convolution, CosineAttention
from .Resamplers import InterpolativeDownsampler, InterpolativeUpsampler, InplaceUpsampler, InplaceDownsampler

'''class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, InputChannels, HiddenChannels, EmbeddingDimension, ChannelsPerHead, rngs):

        self.QKVLayer = BiasedPointwiseConvolutionWithModulation(InputChannels, HiddenChannels * 3, EmbeddingDimension, Centered=True, rngs=rngs)
        self.ProjectionLayer = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True, rngs=rngs)
        self.Heads = HiddenChannels // ChannelsPerHead
        self.Sinks = nnx.Param(jnp.zeros(self.Heads))
        self.RoPE = RotaryPositionEmbedding(embed_dim=HiddenChannels, num_heads=self.Heads, rngs=rngs)

    def __call__(self, x, w, InputGain, ResidualGain, key):
        QKVLayer = lambda y: self.QKVLayer(y, w, Gain=InputGain.reshape(1, -1, 1, 1), key=key)
        ProjectionLayer = lambda y: self.ProjectionLayer(y, Gain=ResidualGain.reshape(-1, 1, 1, 1))
        
        N, C, H, W = x.shape
        RoPE = self.RoPE(H=H, W=W)
        
        return x + CosineAttention(x, self.Heads, QKVLayer, ProjectionLayer, self.Sinks, RoPE, a=8)
'''


class FeedForwardNetwork(nnx.Module):
    def __init__(self, InputChannels, HiddenChannels, EmbeddingDimension, ChannelsPerGroup, KernelSize, Noise, rngs):
        
        if Noise:
            self.LinearLayer1 = NoisyBiasedPointwiseConvolutionWithModulation(InputChannels, HiddenChannels, EmbeddingDimension, Centered=True, rngs=rngs)
        else:
            self.LinearLayer1 = BiasedPointwiseConvolutionWithModulation(InputChannels, HiddenChannels, EmbeddingDimension, Centered=True, rngs=rngs)
        
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, Centered=True, rngs=rngs)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True, rngs=rngs)
        self.NonLinearity = LeakyReLU()
        
    def __call__(self, x, w, InputGain, ResidualGain, key=None):
        y = self.LinearLayer1(x, w, Gain=InputGain.reshape(1, -1, 1, 1), key=key)
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y), Gain=ResidualGain.reshape(-1, 1, 1, 1))
        
        return x + y


class ResidualGroup(nnx.Module):
    def __init__(self, InputChannels, BlockConstructors):
        
        self.Layers = [Block(**Arguments) for Block, Arguments in BlockConstructors]
        self.ParametrizedAlphas = [BoundedParameter(InputChannels) for _ in range(len(self.Layers))]

    def __call__(self, x, w, key=None):
        AccumulatedVariance = jnp.ones([])
        for i, (ParametrizedAlpha, Layer) in enumerate(zip(self.ParametrizedAlphas, self.Layers)):
            if key is not None:
                key = jax.random.fold_in(key, i)
                
            Alpha = ParametrizedAlpha()
            x = Layer(x, w, InputGain=lax.rsqrt(AccumulatedVariance), ResidualGain=Alpha, key=key)
            '''def _call(x):
                return Layer(x, w, InputGain=lax.rsqrt(AccumulatedVariance), ResidualGain=Alpha, key=key)
            x = jax.checkpoint(_call)(x)'''
            AccumulatedVariance = AccumulatedVariance + Alpha * Alpha
        
        return x, AccumulatedVariance



class UpsampleLayer(nnx.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        
        self.FastPath = InputChannels == OutputChannels
        
        if self.FastPath:
            self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        else:
            self.Resampler = InplaceUpsampler(ResamplingFilter)
            self.DuplicationRate = OutputChannels * 4 // InputChannels
        
    def __call__(self, x, Gain):
        x = x * Gain.reshape(1, -1, 1, 1).astype(x.dtype)
        
        if self.FastPath:
            return self.Resampler(x)
        else:
            return self.Resampler(jnp.repeat(x, self.DuplicationRate, axis=1))
        

class DownsampleLayer(nnx.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        
        self.FastPath = InputChannels == OutputChannels
        
        if self.FastPath:
            self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        else:
            self.Resampler = InplaceDownsampler(ResamplingFilter)
            self.ReductionRate = InputChannels * 4 // OutputChannels
        
    def __call__(self, x, Gain):
        x = self.Resampler(x * Gain.reshape(1, -1, 1, 1).astype(x.dtype))
        
        if self.FastPath:
            return x
        else:
            return x.reshape(x.shape[0], -1, self.ReductionRate, x.shape[2], x.shape[3]).mean(axis=2)
    


class GenerativeHead(nnx.Module):
    def __init__(self, InputDimension, OutputChannels, ResamplingFilter, rngs):

        self.LinearLayer1 = Linear(InputDimension + 1, OutputChannels * 2, Centered=True, rngs=rngs)
        self.LinearLayer2 = GenerativeBasis(OutputChannels * 2, rngs=rngs)
        self.NonLinearity = LeakyReLU()
        self.LinearLayer3 = Convolution(OutputChannels * 2, OutputChannels, KernelSize=1, Centered=True, rngs=rngs)
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
    def __call__(self, x):
        x = jnp.concatenate([x, jnp.ones_like(x[:, :1])], axis=1)

        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y))

        return self.Resampler(y)

class DiscriminativeHead(nnx.Module):
    def __init__(self, InputChannels, OutputDimension, ResamplingFilter, rngs):

        self.LinearLayer1 = BiasedPointwiseConvolutionWithModulation(InputChannels, InputChannels * 2, None, Centered=True, rngs=rngs)
        self.NonLinearity = LeakyReLU()
        self.LinearLayer2 = DiscriminativeBasis(InputChannels * 2, rngs=rngs)
        self.LinearLayer3 = Linear(InputChannels * 2, OutputDimension, Centered=True, rngs=rngs)
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)

    def __call__(self, x, Gain):
        y = self.LinearLayer1(self.Resampler(x), None, Gain=Gain.reshape(1, -1, 1, 1))
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y))
        return y


def BuildResidualGroups(WidthPerStage, BlocksPerStage, EmbeddingDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead, Noise, rngs):
    ResidualGroups = []
    for Width, Blocks in zip(WidthPerStage, BlocksPerStage):
        BlockConstructors = []
        for BlockType in Blocks:
            if BlockType == 'FFN':
                BlockConstructors += [(FeedForwardNetwork, dict(InputChannels=Width, HiddenChannels=round(Width * FFNWidthRatio), EmbeddingDimension=EmbeddingDimension, ChannelsPerGroup=ChannelsPerConvolutionGroup, KernelSize=KernelSize, Noise=Noise, rngs=rngs))]
            elif BlockType == 'Attention':
                BlockConstructors += [(MultiHeadSelfAttention, dict(InputChannels=Width, HiddenChannels=round(Width * AttentionWidthRatio), EmbeddingDimension=EmbeddingDimension, ChannelsPerHead=ChannelsPerAttentionHead, rngs=rngs))]
        ResidualGroups += [ResidualGroup(Width, BlockConstructors)]
    return ResidualGroups

class Generator(nnx.Module):
    def __init__(self, NoiseDimension, ModulationDimension, OutputChannels, WidthPerStage, BlocksPerStage, MLPWidthRatio, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, rngs, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):

        ModulationDimension = None
        self.NoiseDimension = NoiseDimension
        self.NumberOfClasses = NumberOfClasses
        ClassEmbeddingDimension = ClassEmbeddingDimension if NumberOfClasses is not None else 0

        self.MainLayers = BuildResidualGroups(WidthPerStage, BlocksPerStage, ModulationDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead, Noise=True, rngs=rngs)
        self.TransitionLayers = [UpsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)]

        self.Head = GenerativeHead(NoiseDimension + ClassEmbeddingDimension, WidthPerStage[0], ResamplingFilter, rngs=rngs)
        self.AggregationLayer = Convolution(WidthPerStage[-1], OutputChannels, KernelSize=1, rngs=rngs)
        self.Gain = nnx.Param(jnp.ones([]))
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension, rngs=rngs)
        
    def __call__(self, x, y=None, key=None):
        x = jnp.concatenate([x, self.EmbeddingLayer(y)], axis=1) if hasattr(self, 'EmbeddingLayer') else x
        x = self.Head(x).astype(jnp.bfloat16)
        w = None
       
        for i, (Layer, Transition) in enumerate(zip(self.MainLayers[:-1], self.TransitionLayers)):
            x, AccumulatedVariance = Layer(x, w, key=jax.random.fold_in(key, i))
            x = Transition(x, Gain=lax.rsqrt(AccumulatedVariance))
                
        x, AccumulatedVariance = self.MainLayers[-1](x, w, key=jax.random.fold_in(key, len(self.MainLayers)-1))

        return self.AggregationLayer(x, Gain=self.Gain.value * lax.rsqrt(AccumulatedVariance).reshape(1, -1, 1, 1))


class Discriminator(nnx.Module):
    def __init__(self, ModulationDimension, InputChannels, WidthPerStage, BlocksPerStage, MLPWidthRatio, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, rngs, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        ModulationDimension = None
        
        self.MainLayers = BuildResidualGroups(WidthPerStage, BlocksPerStage, ModulationDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead, Noise=False, rngs=rngs)
        self.TransitionLayers = [DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)]
        
        self.Head = DiscriminativeHead(WidthPerStage[-1], 1 if NumberOfClasses is None else ClassEmbeddingDimension, ResamplingFilter, rngs=rngs)
        self.ExtractionLayer = Convolution(InputChannels, WidthPerStage[0], KernelSize=1, rngs=rngs)
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension, rngs=rngs)
        
    def __call__(self, x, y=None):
        if hasattr(self, 'EmbeddingLayer'):
            y = self.EmbeddingLayer(y)
        w = None
        x = self.ExtractionLayer(x.astype(jnp.bfloat16))
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x, w)
            x = Transition(x, Gain=lax.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x, w)
        
        x = self.Head(x.astype(jnp.float32), Gain=lax.rsqrt(AccumulatedVariance))
        x = jnp.sum((x * y / jnp.sqrt(y.shape[1])), axis=1, keepdims=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.reshape(x.shape[0])