{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Macro Config Schema",
  "type": "object",
  "properties": {
    "version": {
      "type": "string",
      "description": "Version of the macro configuration schema"
    },
    "dataSources": {
      "type": "object",
      "description": "Data sources configuration",
      "additionalProperties": {
        "anyOf": [
          { "type": "boolean" },
          { "type": "object" }
        ]
      }
    },
    "indicators": {
      "type": "object",
      "description": "Indicators configuration",
      "additionalProperties": true
    },
    "scoring": {
      "type": "object",
      "description": "Scoring configuration",
      "properties": {
        "scoreMap": {
          "type": "object",
          "additionalProperties": true
        },
        "weights": {
          "type": "object",
          "additionalProperties": {
            "type": "number"
          }
        },
        "aggregationMethod": {
          "type": "string"
        }
      },
      "required": ["scoreMap", "weights", "aggregationMethod"],
      "additionalProperties": false
    },
    "biasAssignment": {
      "type": "object",
      "description": "Bias assignment thresholds",
      "properties": {
        "thresholdHigh": { "type": "number" },
        "thresholdLow": { "type": "number" }
      },
      "required": ["thresholdHigh", "thresholdLow"],
      "additionalProperties": false
    },
    "integration": {
      "type": "object",
      "description": "Integration configuration",
      "additionalProperties": true
    },
    "dynamicUpdates": {
      "type": "object",
      "description": "Dynamic updates configuration",
      "additionalProperties": true
    }
  },
  "required": ["version"],
  "additionalProperties": true
}