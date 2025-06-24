
# Agent: Macronator

## Role
A generic macro-calculation agent. Parses food entries intents, computes calories and macros, updates user memory, and returns structured responses.

## Registration & Implementation

This agent is automatically discovered and registered via the decorator:

```python
from framework.agent import BaseAgent, register_agent
from schema.models import Intent, AgentResponse, UserMemory

@register_agent('macronator')
class Macronator(BaseAgent):
    def handle_intent(self, intent: Intent, memory: UserMemory) -> AgentResponse:
        """
        Handles intents with business_type 'macronator'. Calculates macros from provided food items,
        updates memory.macro_totals_today, and returns a friendly summary.
        """
        items = intent.payload.get('items', [])
        # calculate macros (calories, protein_g, carbs_g, fat_g)
        macros = self.calculate_macros(items)
        # update memory
        memory.macro_totals_today = {
            'calories': memory.macro_totals_today.get('calories', 0) + macros['calories'],
            'protein_g': memory.macro_totals_today.get('protein_g', 0) + macros['protein_g'],
            'carbs_g': memory.macro_totals_today.get('carbs_g', 0) + macros['carbs_g'],
            'fat_g': memory.macro_totals_today.get('fat_g', 0) + macros['fat_g'],
        }
        # return response
        message = (
            f"Logged: {macros['calories']} kcal "
            f"– {macros['protein_g']}g protein, "
            f"{macros['carbs_g']}g carbs, "
            f"{macros['fat_g']}g fat."
        )
        return AgentResponse(message=message, updates={'macro_totals_today': memory.macro_totals_today})

    def calculate_macros(self, items):
        # placeholder for actual macro database logic
        # returns dict with 'calories', 'protein_g', 'carbs_g', 'fat_g'
        ...
```

## Contract

- **Intent**  
  ```json
  {
    "user_id": "string",
    "business_type": "macronator",
    "payload": {
       "items": [
         {"name": "string", "quantity_g": number}
       ]
    }
  }
  ```

- **AgentResponse**  
  ```json
  {
    "message": "string",
    "updates": {
       "macro_totals_today": {
          "calories": number,
          "protein_g": number,
          "carbs_g": number,
          "fat_g": number
       }
    }
  }
  ```

## Example Usage

**Input Intent**  
```json
{
  "user_id": "alice",
  "business_type": "macronator",
  "payload": {
    "items": [
      {"name": "oats", "quantity_g": 50},
      {"name": "peanut butter", "quantity_g": 40}
    ]
  }
}
```

**AgentResponse**  
```json
{
  "message": "Logged: 460 kcal – 38g protein, 30g carbs, 20g fat.",
  "updates": {
    "macro_totals_today": {
      "calories": 460,
      "protein_g": 38,
      "carbs_g": 30,
      "fat_g": 20
    }
  }
}
```
