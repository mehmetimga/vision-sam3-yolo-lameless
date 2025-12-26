import { useEffect, useState } from 'react'
import { modelsApi } from '@/api/client'

export default function ModelConfig() {
  const [parameters, setParameters] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    loadParameters()
  }, [])

  const loadParameters = async () => {
    try {
      const data = await modelsApi.getParameters()
      setParameters(data)
    } catch (error) {
      console.error('Failed to load parameters:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await modelsApi.updateParameters(parameters)
      alert('Parameters saved successfully!')
    } catch (error) {
      alert('Failed to save parameters')
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  if (!parameters) {
    return <div className="text-center py-8">Failed to load parameters</div>
  }

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">Model Configuration</h2>

      <div className="space-y-6">
        {['catboost', 'xgboost', 'lightgbm'].map((model) => (
          <div key={model} className="border rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 capitalize">{model} Parameters</h3>
            <div className="grid gap-4 md:grid-cols-2">
              {parameters[model] &&
                Object.entries(parameters[model]).map(([key, value]: [string, any]) => (
                  <div key={key}>
                    <label className="block text-sm font-medium mb-1">{key}</label>
                    <input
                      type="number"
                      step="0.01"
                      value={value}
                      onChange={(e) => {
                        setParameters({
                          ...parameters,
                          [model]: {
                            ...parameters[model],
                            [key]: Number(e.target.value),
                          },
                        })
                      }}
                      className="w-full px-3 py-2 border rounded-md"
                    />
                  </div>
                ))}
            </div>
          </div>
        ))}

        <div className="border rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Ensemble Configuration</h3>
          <div className="grid gap-4 md:grid-cols-3">
            {parameters.ensemble?.weights &&
              Object.entries(parameters.ensemble.weights).map(([model, weight]: [string, any]) => (
                <div key={model}>
                  <label className="block text-sm font-medium mb-1 capitalize">{model} Weight</label>
                  <input
                    type="number"
                    step="0.01"
                    value={weight}
                    onChange={(e) => {
                      setParameters({
                        ...parameters,
                        ensemble: {
                          ...parameters.ensemble,
                          weights: {
                            ...parameters.ensemble.weights,
                            [model]: Number(e.target.value),
                          },
                        },
                      })
                    }}
                    className="w-full px-3 py-2 border rounded-md"
                  />
                </div>
              ))}
          </div>
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="px-6 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Save Parameters'}
        </button>
      </div>
    </div>
  )
}

