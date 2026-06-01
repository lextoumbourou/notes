---
title: Template Method Pattern
date: 2026-05-31 00:00
modified: 2026-05-31 00:00
status: draft
---

The **Template Method Pattern** is a design pattern where a base class defines the fixed skeleton of an operation and calls out to specific steps, or "hooks", that subclasses override to customise behaviour [@gammaDesignPatternsElements1995]. This locks the overall control flow and lifecycle into the base class while letting subclasses supply only the parts that vary.

For example, the base class owns the `export()` skeleton, marks `writeBody()` as a step subclasses must implement, and leaves `afterWrite()` as an optional hook:

```typescript
abstract class DataExporter {
  // Template method — the fixed skeleton, never overridden
  export(): void {
    this.open();
    this.writeBody();    // varies per subclass
    this.afterWrite();   // optional hook
    this.close();
  }

  protected abstract writeBody(): void;  // subclasses MUST implement
  protected afterWrite(): void {}         // hook — no-op by default

  private open(): void { /* ...shared setup... */ }
  private close(): void { /* ...shared teardown... */ }
}

class CsvExporter extends DataExporter {
  protected writeBody(): void {
    // write rows as CSV
  }
}
```
