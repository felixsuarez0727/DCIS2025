import React from 'react';
import { ChevronDown, Brain, Cpu, Layers, GitMerge, Activity, Database } from 'lucide-react';

const ModelArchitecture = () => {
  const modelLayers = [
    { 
      type: 'dense', 
      name: 'Dense', 
      neurons: 256, 
      params: 131584, 
      activation: 'ReLU',
      icon: <Brain className="w-5 h-5 text-blue-600" />,
      color: 'from-blue-500/20 to-indigo-500/20',
      borderColor: 'border-blue-400',
      textColor: 'text-blue-700'
    },
    { 
      type: 'batch_norm', 
      name: 'BatchNormalization', 
      neurons: 256, 
      params: 1024, 
      icon: <Activity className="w-5 h-5 text-emerald-600" />,
      color: 'from-emerald-500/20 to-teal-500/20',
      borderColor: 'border-emerald-400',
      textColor: 'text-emerald-700'
    },
    { 
      type: 'dropout', 
      name: 'Dropout', 
      neurons: 256, 
      params: 0, 
      rate: 0.5,
      icon: <GitMerge className="w-5 h-5 text-orange-600" />,
      color: 'from-orange-500/20 to-amber-500/20',
      borderColor: 'border-orange-400',
      textColor: 'text-orange-700'
    },
    { 
      type: 'dense', 
      name: 'Dense_1', 
      neurons: 128, 
      params: 32896, 
      activation: 'ReLU',
      icon: <Brain className="w-5 h-5 text-blue-600" />,
      color: 'from-blue-500/20 to-indigo-500/20',
      borderColor: 'border-blue-400',
      textColor: 'text-blue-700'
    },
    { 
      type: 'batch_norm', 
      name: 'BatchNormalization_1', 
      neurons: 128, 
      params: 512, 
      icon: <Activity className="w-5 h-5 text-emerald-600" />,
      color: 'from-emerald-500/20 to-teal-500/20',
      borderColor: 'border-emerald-400',
      textColor: 'text-emerald-700'
    },
    { 
      type: 'dropout', 
      name: 'Dropout_1', 
      neurons: 128, 
      params: 0, 
      rate: 0.5,
      icon: <GitMerge className="w-5 h-5 text-orange-600" />,
      color: 'from-orange-500/20 to-amber-500/20',
      borderColor: 'border-orange-400',
      textColor: 'text-orange-700'
    },
    { 
      type: 'dense', 
      name: 'Dense_2', 
      neurons: 7, 
      params: 903, 
      activation: 'Softmax',
      icon: <Layers className="w-5 h-5 text-violet-600" />,
      color: 'from-violet-500/20 to-purple-500/20',
      borderColor: 'border-violet-400',
      textColor: 'text-violet-700'
    }
  ];

  const totalParams = 166919;
  const trainableParams = 166151;
  const nonTrainableParams = 768;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-xl p-6 mb-8 border border-gray-200">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl">
              <Cpu className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Arquitectura del Modelo Neural
              </h1>
              <p className="text-gray-600 mt-1">
                Red neuronal profunda para clasificación de señales
              </p>
            </div>
          </div>
        </div>

        {/* Input Layer */}
        <div className="bg-gradient-to-r from-gray-700 to-slate-800 rounded-xl p-4 mb-6 text-white shadow-lg border border-gray-600">
          <div className="flex items-center gap-3">
            <Database className="w-6 h-6" />
            <div>
              <h3 className="font-semibold text-lg">Input Layer</h3>
              <p className="text-sm text-gray-300">Vector de características (513 dimensiones)</p>
            </div>
          </div>
        </div>

        {/* Layers */}
        <div className="space-y-4">
          {modelLayers.map((layer, index) => (
            <React.Fragment key={index}>
              <div className={`bg-gradient-to-r ${layer.color} rounded-xl p-4 shadow-lg border ${layer.borderColor} backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:shadow-xl`}>
                <div className="grid grid-cols-12 gap-4 items-center">
                  <div className="col-span-4 flex items-center gap-3">
                    <div className="p-2 bg-white rounded-lg shadow-sm">
                      {layer.icon}
                    </div>
                    <div>
                      <h3 className={`font-semibold ${layer.textColor}`}>{layer.name}</h3>
                      <p className="text-sm text-gray-600">Tipo: {layer.type}</p>
                    </div>
                  </div>
                  
                  <div className="col-span-3 text-center">
                    <p className="text-sm text-gray-600">Shape</p>
                    <p className="font-mono font-semibold text-gray-800">(None, {layer.neurons})</p>
                  </div>
                  
                  <div className="col-span-2 text-center">
                    <p className="text-sm text-gray-600">Parámetros</p>
                    <p className="font-semibold text-gray-800">{layer.params.toLocaleString()}</p>
                  </div>
                  
                  <div className="col-span-3 text-right">
                    {layer.activation && (
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-white shadow-sm">
                        <Activity className="w-4 h-4 mr-1.5 text-indigo-600" />
                        {layer.activation}
                      </span>
                    )}
                    {layer.rate && (
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-white shadow-sm">
                        <ChevronDown className="w-4 h-4 mr-1.5 text-orange-600" />
                        {layer.rate * 100}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
              
              {index < modelLayers.length - 1 && (
                <div className="flex justify-center">
                  <div className="w-0.5 h-8 bg-gradient-to-b from-gray-300 to-transparent"></div>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Statistics */}
        <div className="mt-8 grid grid-cols-3 gap-4">
          <div className="bg-white rounded-xl p-4 shadow-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Parámetros</p>
                <p className="text-2xl font-bold text-gray-800">{totalParams.toLocaleString()}</p>
                <p className="text-xs text-gray-500">652.03 KB</p>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <Layers className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-xl p-4 shadow-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Entrenables</p>
                <p className="text-2xl font-bold text-emerald-600">{trainableParams.toLocaleString()}</p>
                <p className="text-xs text-gray-500">649.03 KB</p>
              </div>
              <div className="p-3 bg-emerald-100 rounded-lg">
                <Activity className="w-6 h-6 text-emerald-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-xl p-4 shadow-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">No Entrenables</p>
                <p className="text-2xl font-bold text-orange-600">{nonTrainableParams.toLocaleString()}</p>
                <p className="text-xs text-gray-500">3.00 KB</p>
              </div>
              <div className="p-3 bg-orange-100 rounded-lg">
                <GitMerge className="w-6 h-6 text-orange-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Model Summary */}
        <div className="mt-8 bg-white rounded-xl p-6 shadow-lg border border-gray-200">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Resumen del Modelo</h2>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <p className="text-sm text-gray-600">Arquitectura</p>
              <p className="font-semibold text-gray-800">Sequential Neural Network</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-600">Caso de Uso</p>
              <p className="font-semibold text-gray-800">Clasificación de Señales de Telecomunicaciones</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-600">Clases de Salida</p>
              <p className="font-semibold text-gray-800">7 tipos de señales</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-600">Función de Pérdida</p>
              <p className="font-semibold text-gray-800">Categorical Crossentropy</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelArchitecture;