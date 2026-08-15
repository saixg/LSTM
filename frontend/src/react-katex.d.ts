declare module 'react-katex' {
  import * as React from 'react';
  export interface MathProps {
    math?: string;
    children?: React.ReactNode;
    renderError?: (error: Error | TypeError) => React.ReactNode;
    as?: string;
  }
  export class InlineMath extends React.Component<MathProps> {}
  export class BlockMath extends React.Component<MathProps> {}
}
