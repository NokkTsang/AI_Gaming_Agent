# AI Gaming Agent 🎮

A comprehensive AI gaming agent framework built on top of **smolagents** to create intelligent agents capable of playing and mastering various types of games, from simple board games to complex universal gaming environments.

## Technology Stack

**Core Framework**: [SmolAgents](https://github.com/huggingface/smolagents) - A powerful agent framework for building AI applications

## Architecture

```mermaid
---
config:
  layout: dagre
---
flowchart TB
 subgraph s1["**Data Flow**"]
        A["Screen Input"]
        B["Information<br>Gathering"]
        H["UI Automation"]
        I["Keyboard | Mouse<br>Output"]
        J["Computer Environment"]
  end
 subgraph s2["**Agent Module**"]
        C["Memory"]
        D["Self-Reflection"]
        E["Task Inference"]
        F["Skill Curation"]
        G["Action Planning"]
  end
    A --> B
    B --> C
    C --> D & E
    D --> C
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> A
    style s1 fill:#FFE0B2
    style s2 fill:#C8E6C9
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hugging Face** for the amazing smolagents framework
