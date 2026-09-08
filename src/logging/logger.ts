export interface LoggerProtocol {
  log(message: string): void;
}

export class StdOutLogger implements LoggerProtocol {
  log(message: string): void {
    console.log(message);
  }
}

export class Logger implements LoggerProtocol {
  readonly messages: string[] = [];

  log(...args: unknown[]): void {
    this.messages.push(args.map((arg) => String(arg)).join(' '));
  }
}
