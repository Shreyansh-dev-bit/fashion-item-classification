import React, { useState, useRef } from 'react'
import {
  UploadCloud,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  FileImage,
  RefreshCw,
  Info,
  Medal,
  SlidersHorizontal,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { uploadPredictImage } from '../services/api'
import { PredictResponse, SamplePredictionItem } from '../types/api'

interface RealPredictionPageProps {
  samples: SamplePredictionItem[]
}

export function RealPredictionPage({ samples }: RealPredictionPageProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [invertOption, setInvertOption] = useState<boolean | undefined>(undefined)
  const [isDragging, setIsDragging] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0])
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0])
    }
  }

  const processSelectedFile = (file: File) => {
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    const validExtensions = ['.png', '.jpg', '.jpeg']

    if (!validExtensions.includes(ext)) {
      setError(
        `Invalid file type "${file.name}". Only PNG, JPG, and JPEG images are supported by the CNN inference pipeline.`
      )
      setSelectedFile(null)
      setPreviewUrl(null)
      setPrediction(null)
      return
    }

    setSelectedFile(file)
    setError(null)
    setPrediction(null)

    // Create object URL for preview
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)

    // Execute real CNN prediction
    executeInference(file, invertOption)
  }

  const executeInference = async (file: File, invert?: boolean) => {
    try {
      setLoading(true)
      setError(null)
      const res = await uploadPredictImage(file, invert)
      setPrediction(res)
    } catch (err: any) {
      setError(err.message || 'Model prediction failed. Please ensure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleSelectSample = async (sample: SamplePredictionItem) => {
    try {
      setLoading(true)
      setError(null)
      const res = await fetch(sample.image_base64)
      const blob = await res.blob()
      const file = new File([blob], `sample_${sample.test_index}_${sample.true_label_name}.png`, {
        type: 'image/png',
      })
      setSelectedFile(file)
      setPreviewUrl(sample.image_base64)
      await executeInference(file, false)
    } catch (err: any) {
      setError(err.message || 'Failed to process sample image')
    } finally {
      setLoading(false)
    }
  }

  // Sorted Top 3 Predictions
  const topThree = prediction
    ? [...prediction.probabilities].sort((a, b) => b.probability - a.probability).slice(0, 3)
    : []

  // Full 10-class probability chart data
  const chartData = prediction
    ? prediction.probabilities.map((item) => ({
        name: item.class_name,
        percentage: item.percentage,
        isTop: item.class_id === prediction.predicted_class_id,
      }))
    : []

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Title */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100 mb-2">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Real CNN Inference Service</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Live Fashion Item Classification
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Upload any apparel image (PNG, JPG, JPEG) to classify it using the trained Convolutional Neural Network.
        </p>
      </div>

      {/* Distribution Shift & Benchmark Reality Callout */}
      <div className="p-5 rounded-2xl bg-amber-50/70 border border-amber-200/80 text-amber-950 flex items-start gap-3.5 shadow-2xs">
        <div className="p-2 rounded-xl bg-amber-100 text-amber-700 shrink-0 mt-0.5">
          <Info className="w-5 h-5" />
        </div>
        <div className="text-xs sm:text-sm space-y-1">
          <h4 className="font-bold text-amber-950">
            Important Note on Benchmark Distribution & Color Photos
          </h4>
          <p className="text-amber-900 leading-relaxed">
            This deep learning model was trained strictly on <strong>28×28 grayscale</strong> Fashion-MNIST images featuring centered clothing on dark backgrounds. 
            When uploading ordinary full-color retail photos, prediction accuracy may be lower because real-world product photos exhibit a <strong>distribution shift</strong> (RGB color variations, diverse lighting, complex retail backgrounds, and multi-angle perspectives) compared to the normalized 28×28 grayscale benchmark.
          </p>
        </div>
      </div>

      {/* Main Grid: Upload vs Result */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload Box & Quick Test Presets (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Upload Zone */}
          <div className="p-6 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
            <h3 className="text-sm font-bold text-slate-900 mb-1">Image Ingestion Box</h3>
            <p className="text-xs text-slate-500 mb-4">Drag and drop or browse local files (PNG, JPG, JPEG)</p>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".png,.jpg,.jpeg,image/png,image/jpeg"
              className="hidden"
            />

            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[190px] ${
                isDragging
                  ? 'border-indigo-600 bg-indigo-50 shadow-md scale-102'
                  : 'border-slate-300 hover:border-indigo-500 bg-slate-50/50 hover:bg-indigo-50/20'
              }`}
            >
              {previewUrl ? (
                <div className="flex flex-col items-center">
                  <div className="w-24 h-24 rounded-2xl overflow-hidden border-2 border-indigo-500/40 bg-black shadow-md mb-3 flex items-center justify-center p-1">
                    <img
                      src={previewUrl}
                      alt="Uploaded apparel preview"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <span className="text-xs font-bold text-slate-900 truncate max-w-[220px]">
                    {selectedFile?.name || 'Selected Image'}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                    {selectedFile?.size ? `${(selectedFile.size / 1024).toFixed(1)} KB` : 'Preset sample'}
                  </span>
                  <span className="text-[11px] text-indigo-600 mt-2 font-semibold hover:underline">
                    Click or drop another file to change
                  </span>
                </div>
              ) : (
                <>
                  <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center mb-3">
                    <UploadCloud className="w-6 h-6" />
                  </div>
                  <span className="text-xs font-bold text-slate-800">
                    Drag and drop your apparel image here
                  </span>
                  <span className="text-[11px] text-indigo-600 font-semibold mt-1">
                    or click to browse local files
                  </span>
                  <span className="text-[10px] text-slate-400 mt-2">PNG, JPG, or JPEG</span>
                </>
              )}
            </div>

            {/* Inversion Setting */}
            <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs text-slate-600">
                <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
                <span>Background Adaptation:</span>
              </div>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => {
                    setInvertOption(undefined)
                    if (selectedFile) executeInference(selectedFile, undefined)
                  }}
                  className={`px-2.5 py-1 rounded-lg text-[10px] font-semibold transition-all ${
                    invertOption === undefined
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  Auto
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setInvertOption(true)
                    if (selectedFile) executeInference(selectedFile, true)
                  }}
                  className={`px-2.5 py-1 rounded-lg text-[10px] font-semibold transition-all ${
                    invertOption === true
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                  title="Invert bright white background to dark Fashion-MNIST format"
                >
                  Invert
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setInvertOption(false)
                    if (selectedFile) executeInference(selectedFile, false)
                  }}
                  className={`px-2.5 py-1 rounded-lg text-[10px] font-semibold transition-all ${
                    invertOption === false
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                  title="Preserve original background"
                >
                  Standard
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <span className="leading-snug">{error}</span>
              </div>
            )}
          </div>

          {/* Quick-Test Presets from real Fashion-MNIST test set */}
          <div className="p-6 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">
                One-Click Test Presets
              </h3>
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
                Real Test Set
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-3.5">
              Click any authentic 28×28 sample below to test inference instantly:
            </p>

            <div className="grid grid-cols-5 gap-2">
              {samples.slice(0, 10).map((sample) => (
                <button
                  key={sample.test_index}
                  type="button"
                  onClick={() => handleSelectSample(sample)}
                  className="p-1.5 rounded-xl border border-slate-200 hover:border-indigo-500 hover:bg-indigo-50/40 transition-all flex flex-col items-center group text-center cursor-pointer shadow-2xs"
                  title={`${sample.true_label_name} (Index #${sample.test_index})`}
                >
                  <img
                    src={sample.image_base64}
                    alt={sample.true_label_name}
                    className="w-10 h-10 object-contain rounded bg-black"
                  />
                  <span className="text-[9px] font-bold text-slate-700 truncate w-full mt-1 group-hover:text-indigo-900">
                    {sample.true_label_name.split('/')[0]}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results, Sorted Top 3, & Distribution (7 cols) */}
        <div className="lg:col-span-7">
          <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs min-h-[500px] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-base font-bold text-slate-900">CNN Model Inference Output</h3>
                {prediction && (
                  <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Real Forward Pass Completed</span>
                  </span>
                )}
              </div>

              {loading ? (
                <div className="py-28 text-center">
                  <RefreshCw className="w-9 h-9 text-indigo-600 animate-spin mx-auto mb-3" />
                  <p className="text-sm font-semibold text-slate-800">
                    Running CNN Neural Network Inference...
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Preprocessing image to (1, 28, 28, 1) and computing softmax distribution
                  </p>
                </div>
              ) : prediction ? (
                <div className="space-y-6">
                  {/* Top Predicted Category Banner */}
                  <div className="p-5 rounded-2xl bg-gradient-to-r from-indigo-50 via-violet-50/40 to-transparent border border-indigo-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                      <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-600">
                        Top Predicted Category
                      </span>
                      <div className="text-2xl sm:text-3xl font-black text-slate-900 mt-0.5">
                        {prediction.predicted_class_name}
                      </div>
                      <span className="text-xs text-slate-500 font-mono">
                        Class Index: #{prediction.predicted_class_id}
                      </span>
                    </div>

                    <div className="sm:text-right">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                        Confidence Score
                      </span>
                      <div className="text-2xl sm:text-3xl font-black text-indigo-600 font-mono mt-0.5">
                        {prediction.confidence_percentage}%
                      </div>
                      <span className="text-[10px] text-slate-400">Softmax output unit</span>
                    </div>
                  </div>

                  {/* Sorted Top-3 Ranked Predictions */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                      <Medal className="w-4 h-4 text-amber-500" />
                      <span>Sorted Top 3 Probabilities</span>
                    </h4>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {topThree.map((item, idx) => {
                        const rankColors = [
                          'bg-indigo-50 border-indigo-200 text-indigo-950',
                          'bg-slate-50 border-slate-200 text-slate-800',
                          'bg-slate-50 border-slate-200 text-slate-700',
                        ]
                        const badgeColors = [
                          'bg-indigo-600 text-white',
                          'bg-slate-400 text-white',
                          'bg-slate-300 text-slate-700',
                        ]

                        return (
                          <div
                            key={item.class_id}
                            className={`p-3.5 rounded-xl border ${rankColors[idx]} flex flex-col justify-between`}
                          >
                            <div className="flex items-center justify-between mb-1.5">
                              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${badgeColors[idx]}`}>
                                #{idx + 1} Rank
                              </span>
                              <span className="text-[10px] font-mono text-slate-400 font-semibold">
                                #{item.class_id}
                              </span>
                            </div>

                            <div className="font-bold text-sm truncate my-1">
                              {item.class_name}
                            </div>

                            <div className="flex items-end justify-between mt-1">
                              <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden mr-2">
                                <div
                                  className={`h-full rounded-full ${idx === 0 ? 'bg-indigo-600' : 'bg-slate-400'}`}
                                  style={{ width: `${item.percentage}%` }}
                                />
                              </div>
                              <span className="font-mono text-xs font-bold shrink-0">
                                {item.percentage}%
                              </span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  {/* 10-Class Horizontal Probability Chart */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                      Full 10-Class Output Vector
                    </h4>
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={chartData}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                        >
                          <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 10, fill: '#64748B' }} />
                          <YAxis
                            dataKey="name"
                            type="category"
                            tick={{ fontSize: 11, fill: '#334155', fontWeight: 600 }}
                            width={75}
                          />
                          <Tooltip
                            formatter={(value: any) => [`${value}%`, 'Probability']}
                            contentStyle={{
                              backgroundColor: '#0F172A',
                              borderRadius: '10px',
                              border: 'none',
                              color: '#F8FAFC',
                              fontSize: '12px',
                            }}
                          />
                          <Bar dataKey="percentage" radius={[0, 6, 6, 0]}>
                            {chartData.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={entry.isTop ? '#4F46E5' : '#CBD5E1'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Preprocessing Inspector Footer */}
                  <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100 flex flex-wrap items-center justify-between text-xs text-slate-600 gap-2">
                    <div className="flex items-center gap-1.5">
                      <FileImage className="w-4 h-4 text-slate-400" />
                      <span>Original: {prediction.input_image_details.original_size.join('×')} px</span>
                    </div>
                    <span className="text-slate-300">&bull;</span>
                    <span>Grayscale 28×28</span>
                    <span className="text-slate-300">&bull;</span>
                    <span>Normalized [0, 1]</span>
                    <span className="text-slate-300">&bull;</span>
                    <span className="font-mono text-indigo-700 font-semibold">
                      {prediction.input_image_details.inverted_for_dark_background
                        ? 'Auto-Inverted (Light Background Detected)'
                        : 'Standard (Dark Background)'}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="py-28 text-center text-slate-400">
                  <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto mb-3">
                    <Sparkles className="w-7 h-7" />
                  </div>
                  <h4 className="text-sm font-semibold text-slate-700">No Image Classified Yet</h4>
                  <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1">
                    Upload an image or pick a test preset from the left to execute the real CNN forward pass.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
