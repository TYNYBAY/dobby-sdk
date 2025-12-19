---
trigger: always_on
description: Coding standards, folder structure, and conventions for Next.js frontend projects
---

# Frontend Coding Standards & Conventions

Use this prompt as a reference for all AI agents and developers working on this Next.js frontend codebase. Follow these standards to maintain consistency across the project.

> **Note:** All packages use the **latest stable versions**. Do not hardcode version numbers.

---

## Technology Stack

### Core Framework
- **Next.js** (latest stable) with App Router (Turbopack enabled via `next dev --turbo`)
- **React** (latest stable)
- **TypeScript** (latest stable) with strict mode enabled

### Styling
- **Tailwind CSS** (latest stable) with PostCSS
- **CSS Variables** for theming (light/dark mode support)
- **class-variance-authority (CVA)** for component variants
- **tailwind-merge** + **clsx** via `cn()` helper for conditional classes

### UI Components
- **shadcn/ui** (default style, Zinc base color, CSS variables enabled)
- **Radix UI** primitives for accessible components
- **Lucide React** for icons
- **Framer Motion** for animations

### State Management & Data Fetching
- **SWR** for data fetching and caching
- **React Context API** for global state (auth, themes, etc.)
- **useState/useCallback/useRef** hooks for local state

### API & Backend Integration
- **Custom API client** with timeout handling and Bearer token auth
- **Zod** for runtime validation

### Code Quality
- **Biome** for linting and formatting (extends `ultracite` config)
- **ultracite** as the base linting preset
- **Playwright** for E2E testing
- **pnpm** as package manager

---

## Folder Structure

```
web/
├── app/                          # Next.js App Router
│   ├── (auth)/                   # Route group: authentication pages
│   ├── (main)/                   # Route group: main feature pages
│   │   ├── api/                  # Route handlers (API routes)
│   │   ├── layout.tsx            # Nested layout
│   │   └── page.tsx              # Home/landing page
│   ├── globals.css               # Global styles & Tailwind config
│   ├── layout.tsx                # Root layout with providers
│   └── favicon.ico
│
├── components/                   # React components
│   ├── ui/                       # shadcn/ui primitives (button, input, etc.)
│   ├── elements/                 # Domain-specific UI elements
│   └── *.tsx                     # Feature-level components
│
├── contexts/                     # React Context providers
│
├── hooks/                        # Custom React hooks
│   └── use-*.ts                  # Hook files
│
├── lib/                          # Utilities and business logic
│   ├── api/                      # API client functions
│   │   ├── client.ts             # Centralized API client
│   │   └── *.ts                  # Domain-specific API modules
│   ├── auth/                     # Authentication logic
│   ├── types/                    # Shared TypeScript types
│   ├── types.ts                  # Core type definitions
│   ├── utils.ts                  # General utilities (cn, fetcher, etc.)
│   ├── errors.ts                 # Custom error classes
│   └── constants.ts              # App constants
│
├── public/                       # Static assets
├── tests/                        # Playwright E2E tests
│
├── biome.jsonc                   # Biome linter config
├── components.json               # shadcn/ui configuration
├── next.config.ts                # Next.js configuration
├── tsconfig.json                 # TypeScript configuration
└── package.json                  # Dependencies and scripts
```

---

## Coding Conventions

### File Naming
- **Components**: `kebab-case.tsx` (e.g., `chat-header.tsx`, `user-profile.tsx`)
- **Hooks**: `use-kebab-case.ts` (e.g., `use-auth.ts`, `use-form.ts`)
- **Utilities**: `kebab-case.ts` (e.g., `utils.ts`, `client.ts`)
- **Types**: `types.ts` or grouped in `lib/types/` directory

### Component Patterns

#### Client Components
```tsx
"use client";

import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface MyComponentProps {
  title: string;
  variant?: "default" | "outline";
  onAction?: () => void;
}

export function MyComponent({
  title,
  variant = "default",
  onAction,
}: MyComponentProps) {
  const [isActive, setIsActive] = useState(false);

  const handleClick = useCallback(() => {
    setIsActive((prev) => !prev);
    onAction?.();
  }, [onAction]);

  return (
    <div className={cn("p-4 rounded-lg", isActive && "bg-accent")}>
      <h2 className="text-lg font-semibold">{title}</h2>
      <Button variant={variant} onClick={handleClick}>
        Toggle
      </Button>
    </div>
  );
}
```

### Import Order
1. React/Next.js imports
2. Third-party libraries
3. Internal components (`@/components/...`)
4. Hooks (`@/hooks/...`)
5. Utilities/lib (`@/lib/...`)
6. Types
7. Relative imports

```tsx
"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/header";
import { useAuth } from "@/hooks/use-auth";
import { fetchData } from "@/lib/api/data";
import type { DataItem } from "@/lib/types";
```

### TypeScript Patterns

#### Interface Definitions
```tsx
// Prefer interfaces for public APIs
interface ComponentProps {
  id: string;
  title: string;
  isActive?: boolean;
  onToggle?: (id: string) => void;
}

// Use type for unions and complex types
type Status = "ready" | "loading" | "error";
```

#### Generic Functions
```tsx
export async function apiClient<T>(
  path: string,
  token: string,
  options: RequestOptions = {}
): Promise<T> {
  // Implementation
}
```

### Styling with Tailwind

#### Using the `cn()` Helper
```tsx
import { cn } from "@/lib/utils";

// Conditional classes
<div className={cn(
  "p-4 rounded-lg border",
  isActive && "border-primary bg-accent",
  variant === "outline" && "bg-transparent"
)} />
```

#### CVA for Component Variants
```tsx
import { cva, type VariantProps } from "class-variance-authority";

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md text-sm font-medium",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        outline: "border border-input bg-background hover:bg-accent",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 px-3",
        lg: "h-11 px-8",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);
```

### Custom Hooks Pattern
```tsx
"use client";

import { useCallback, useState } from "react";

interface UseFeatureOptions {
  initialValue?: string;
  onComplete?: () => void;
}

export function useFeature({ initialValue = "", onComplete }: UseFeatureOptions) {
  const [value, setValue] = useState(initialValue);
  const [status, setStatus] = useState<"idle" | "loading" | "success">("idle");

  const execute = useCallback(async () => {
    setStatus("loading");
    try {
      // async operation
      setStatus("success");
      onComplete?.();
    } catch (error) {
      setStatus("idle");
      throw error;
    }
  }, [onComplete]);

  return { value, setValue, status, execute };
}
```

### Context Provider Pattern
```tsx
"use client";

import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from "react";

type ContextType = {
  value: string;
  setValue: (v: string) => void;
};

const MyContext = createContext<ContextType | null>(null);

export function MyProvider({ children }: { children: ReactNode }) {
  const [value, setValueState] = useState("");
  
  const setValue = useCallback((v: string) => {
    setValueState(v);
  }, []);

  return (
    <MyContext.Provider value={{ value, setValue }}>
      {children}
    </MyContext.Provider>
  );
}

export function useMyContext() {
  const context = useContext(MyContext);
  if (!context) {
    throw new Error("useMyContext must be used within MyProvider");
  }
  return context;
}
```

### API Client Pattern
```tsx
import { apiClient } from "./client";

export interface Item {
  id: string;
  title: string;
  createdAt: string;
}

export async function getItems(token: string): Promise<Item[]> {
  return apiClient<Item[]>("/api/items", token, { method: "GET" });
}

export async function createItem(
  token: string, 
  data: { title: string }
): Promise<Item> {
  return apiClient<Item>("/api/items", token, {
    method: "POST",
    body: data,
  });
}
```

### Error Handling
```tsx
import { AppError, type ErrorCode } from "@/lib/errors";

try {
  const response = await fetch(url);
  if (!response.ok) {
    const { code, cause } = await response.json();
    throw new AppError(code as ErrorCode, cause);
  }
} catch (error) {
  if (error instanceof AppError) {
    toast({ type: "error", description: error.message });
  }
  throw error;
}
```

---

## Design System Tokens (CSS Variables)

```css
/* Light mode (default) */
--background: hsl(0 0% 100%);
--foreground: hsl(240 10% 3.9%);
--primary: hsl(240 5.9% 10%);
--primary-foreground: hsl(0 0% 98%);
--secondary: hsl(240 4.8% 95.9%);
--muted: hsl(240 4.8% 95.9%);
--muted-foreground: hsl(240 3.8% 46.1%);
--accent: hsl(240 4.8% 95.9%);
--destructive: hsl(0 84.2% 60.2%);
--border: hsl(240 5.9% 90%);
--radius: 0.5rem;

/* Dark mode (.dark class) */
--background: hsl(240 10% 3.9%);
--foreground: hsl(0 0% 98%);
/* ... dark variants */
```

---

## NPM Scripts

```bash
pnpm dev        # Start dev server with Turbopack
pnpm build      # Production build
pnpm start      # Start production server
pnpm lint       # Run Biome linter (via ultracite)
pnpm format     # Fix linting issues
pnpm test       # Run Playwright tests
```

---

## Path Aliases

| Alias | Path |
|-------|------|
| `@/*` | `./` (project root) |
| `@/components` | `./components` |
| `@/components/ui` | `./components/ui` |
| `@/lib` | `./lib` |
| `@/hooks` | `./hooks` |

---

## Key Dependencies Reference

| Package | Purpose |
|---------|---------|
| `next` | React framework with App Router |
| `react` / `react-dom` | UI library |
| `typescript` | Type safety |
| `tailwindcss` | Utility-first CSS |
| `@biomejs/biome` | Fast linter and formatter |
| `ultracite` | Biome preset with opinionated rules |
| `@radix-ui/*` | Accessible UI primitives |
| `class-variance-authority` | Component variant system |
| `clsx` + `tailwind-merge` | Class name utilities |
| `lucide-react` | Icon library |
| `framer-motion` | Animations |
| `swr` | Data fetching/caching |
| `zod` | Runtime validation |
| `sonner` | Toast notifications |
| `next-themes` | Theme management |
| `@playwright/test` | E2E testing |

---

## Quick Checklist for New Components

- [ ] Use appropriate file naming (`kebab-case.tsx`)
- [ ] Add `"use client"` directive if using hooks/state
- [ ] Define TypeScript interface for props
- [ ] Use `cn()` helper for conditional classes
- [ ] Import from path aliases (`@/components`, `@/lib`, etc.)
- [ ] Follow import order convention
- [ ] Use semantic HTML elements
- [ ] Add proper TypeScript types (avoid `any`)
- [ ] Use design tokens from CSS variables