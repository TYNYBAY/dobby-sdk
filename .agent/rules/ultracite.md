---
trigger: model_decision
description: Ultracite Rules - AI-Ready Formatter and Linter
globs: "**/*.{ts,tsx,js,jsx}"
---

# Ultracite Linting Rules

Ultracite enforces strict type safety, accessibility standards, and consistent code quality for JavaScript/TypeScript projects using Biome's lightning-fast formatter and linter.

## Key Principles
- Zero configuration required
- Subsecond performance
- Maximum type safety
- AI-friendly code generation

## Before Writing Code
1. Analyze existing patterns in the codebase
2. Consider edge cases and error scenarios
3. Follow the rules below strictly
4. Validate accessibility requirements

---

## Accessibility (a11y)

- Don't use accessKey attribute on any HTML element
- Don't set aria-hidden="true" on focusable elements
- Don't add ARIA roles, states, and properties to elements that don't support them
- Don't use distracting elements like marquee or blink
- Only use the scope prop on th elements
- Don't assign non-interactive ARIA roles to interactive HTML elements
- Make sure label elements have text content and are associated with an input
- Don't assign interactive ARIA roles to non-interactive HTML elements
- Don't assign tabIndex to non-interactive HTML elements
- Don't use positive integers for tabIndex property
- Don't include "image", "picture", or "photo" in img alt prop
- Don't use explicit role property that's the same as the implicit/default role
- Make static elements with click handlers use a valid role attribute
- Always include a title element for SVG elements
- Give all elements requiring alt text meaningful information for screen readers
- Make sure anchors have content that's accessible to screen readers
- Assign tabIndex to non-interactive HTML elements with aria-activedescendant
- Include all required ARIA attributes for elements with ARIA roles
- Make sure ARIA properties are valid for the element's supported roles
- Always include a type attribute for button elements
- Make elements with interactive roles and handlers focusable
- Give heading elements content that's accessible to screen readers
- Always include a lang attribute on the html element
- Always include a title attribute for iframe elements
- Accompany onClick with at least one of: onKeyUp, onKeyDown, or onKeyPress
- Accompany onMouseOver/onMouseOut with onFocus/onBlur
- Include caption tracks for audio and video elements
- Use semantic elements instead of role attributes in JSX
- Make sure all anchors are valid and navigable
- Ensure all ARIA properties are valid
- Use valid, non-abstract ARIA roles
- Use valid ARIA state and property values
- Use valid values for the autocomplete attribute on input elements
- Use correct ISO language/country codes for the lang attribute

---

## TypeScript Best Practices

- Don't use TypeScript enums - use const objects or union types instead
- Don't export imported variables
- Don't add type annotations to variables initialized with literal expressions
- Don't use TypeScript namespaces - use ES modules
- Don't use non-null assertions with the ! postfix operator
- Don't use parameter properties in class constructors
- Use as const instead of literal types and type annotations
- Use either T[] or Array consistently throughout the codebase
- Initialize each enum member value explicitly if you must use enums
- Use export type for type exports
- Use import type for type imports
- Make sure all enum members are literal values
- Don't use TypeScript const enum
- Don't declare empty interfaces
- Don't let variables evolve into any type through reassignments
- Don't use the any type - use unknown or proper types
- Don't misuse the non-null assertion operator
- Don't use implicit any type on variable declarations
- Don't merge interfaces and classes unsafely
- Don't use overload signatures that aren't next to each other
- Use the namespace keyword instead of module keyword if you must use namespaces

---

## React and JSX Best Practices

- Don't use the return value of React.render
- Make sure all dependencies are correctly specified in React hooks
- Make sure all React hooks are called from the top level of component functions
- Don't forget key props in iterators and collection literals
- Don't define React components inside other components
- Don't use event handlers on non-interactive elements
- Don't assign to React component props
- Don't use both children and dangerouslySetInnerHTML on the same element
- Don't use dangerous JSX props
- Don't use Array index in keys
- Don't insert comments as text nodes
- Don't assign JSX properties multiple times
- Don't add extra closing tags for components without children
- Use short fragment syntax instead of Fragment component
- Watch out for possible wrong semicolons inside JSX elements

---

## Next.js Specific Rules

- Don't use img elements - use next/image instead
- Don't use head elements - use next/head instead
- Don't import next/document outside of pages/_document.jsx
- Don't use the next/head module in pages/_document.js

---

## Code Quality and Complexity

- Don't use consecutive spaces in regular expression literals
- Don't use the arguments object - use rest parameters
- Don't use primitive type aliases or misleading types
- Don't use the comma operator
- Don't use empty type parameters in type aliases and interfaces
- Don't write functions that exceed cognitive complexity limits
- Don't nest describe blocks too deeply in test files
- Don't use unnecessary boolean casts
- Don't use unnecessary callbacks with flatMap
- Use for-of statements instead of Array.forEach
- Don't create classes that only have static members
- Don't use this and super in static contexts
- Don't use unnecessary catch clauses
- Don't use unnecessary constructors
- Don't use unnecessary continue statements
- Don't export empty modules
- Don't use unnecessary escape sequences in regex
- Don't use unnecessary fragments
- Don't use unnecessary labels
- Don't use unnecessary nested block statements
- Don't rename imports/exports to the same name
- Don't use unnecessary string concatenation
- Don't use String.raw when there are no escape sequences
- Don't use useless case statements in switch
- Don't use ternary operators when simpler alternatives exist
- Don't use useless this aliasing
- Don't use any or unknown as type constraints
- Don't initialize variables to undefined
- Don't use void operators
- Use arrow functions instead of function expressions
- Use Date.now to get milliseconds since Unix Epoch
- Use flatMap instead of map then flat
- Use literal property access instead of computed when possible
- Use concise optional chaining instead of chained logical expressions
- Use regex literals instead of RegExp constructor when possible
- Remove redundant terms from logical expressions
- Use while loops instead of for when you don't need initializer/update

---

## Correctness and Safety

- Don't assign a value to itself
- Don't return a value from a setter
- Don't use lexical declarations in switch clauses
- Don't use variables that haven't been declared
- Don't write unreachable code
- Make sure super is called exactly once before this access in constructors
- Don't use control flow statements in finally blocks
- Don't use optional chaining where undefined isn't allowed
- Don't have unused function parameters
- Don't have unused imports
- Don't have unused labels
- Don't have unused private class members
- Don't have unused variables
- Make sure void elements don't have children
- Don't return a value from void functions
- Use isNaN when checking for NaN
- Make sure loop update clauses move counter correctly
- Make sure typeof is compared to valid values
- Make sure generators contain yield
- Don't use await inside loops
- Don't use bitwise operators
- Don't use expressions where operation doesn't change value
- Handle Promises appropriately - no floating promises
- Don't use __dirname/__filename in global scope
- Prevent import cycles
- Don't hardcode sensitive data like API keys and tokens
- Don't let variable declarations shadow outer scope variables
- Don't use ts-ignore directive
- Make sure getters and setters are adjacent in classes
- Make switch-case statements exhaustive
- Use preconnect when using Google Fonts
- Use Array.indexOf instead of findIndex for simple lookups
- Use numeric separators in large numbers
- Use object spread instead of Object.assign
- Always use the radix argument with parseInt

---

## Style and Consistency

- Don't use global eval
- Don't use callbacks in async tests and hooks
- Don't use negation in if statements with else clauses
- Don't use nested ternary expressions
- Don't reassign function parameters
- Use String.slice instead of substr and substring
- Don't use template literals without interpolation
- Don't use else when if block breaks early
- Don't use yoda expressions
- Don't use Array constructors
- Use at method instead of integer index access
- Follow curly brace conventions
- Use else if instead of nested if in else clauses
- Use const for variables only assigned once
- Put default/optional parameters last
- Include a default clause in switch statements
- Use exponentiation operator instead of Math.pow
- Use node: protocol for Node.js builtin modules
- Use Number.isFinite instead of global isFinite
- Use Number.isNaN instead of global isNaN
- Use assignment operator shorthand where possible
- Use template literals over string concatenation
- Use new Error when throwing errors
- Don't throw non-Error values
- Use strict equality operators
- Don't use duplicate case labels
- Don't use duplicate class members
- Don't use duplicate conditions in if-else chains
- Don't use duplicate object keys
- Don't use duplicate function parameter names
- Don't use empty block statements
- Don't let switch clauses fall through
- Don't reassign function declarations
- Don't assign to imported bindings
- Don't use var - use const or let
- Make sure async functions use await
- Make default clauses come last in switch
- Pass a message when creating Error objects
- Make sure getters return a value
- Use Array.isArray instead of instanceof Array

---

## Security

- Don't hardcode API keys, tokens, or secrets
- Don't use target="_blank" without rel="noopener"
- Don't use ts-ignore to bypass type checking
- Don't assign directly to document.cookie

---

## Testing Best Practices

- Don't use export or module.exports in test files
- Don't use focused tests with only
- Make sure assertions are inside it function calls
- Don't use disabled tests with skip
- Don't have duplicate hooks in describe blocks

---

## Commands

npx ultracite init   - Initialize Ultracite in your project
npx ultracite fix    - Format and fix code automatically
npx ultracite check  - Check for issues without fixing


## Example: Error Handling
```typescript
// ✅ Good: Comprehensive error handling
try {
  const result = await fetchData();
  return { success: true, data: result };
} catch (error) {
  console.error('API call failed:', error);
  return { success: false, error: error.message };
}

// ❌ Bad: Swallowing errors
try {
  return await fetchData();
} catch (e) {
  console.log(e);
}
```
