import { describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { Image, _guess_media_type } from '../../src/image.js';
import { InstructionProposalSignature } from '../../src/instruction_proposal.js';

const TINY_PNG_B64 = 'iVBORw0KGgo=';

describe('Image', () => {
  it('converts URL images to OpenAI content parts', () => {
    const img = new Image({ url: 'https://example.com/photo.jpg' });

    expect(img.to_openai_content_part()).toEqual({
      type: 'image_url',
      image_url: { url: 'https://example.com/photo.jpg' },
    });
  });

  it('converts base64 images to data URIs and JSON shim sentinels', () => {
    const img = new Image({ base64_data: TINY_PNG_B64, media_type: 'image/png' });

    expect(img.to_openai_content_part()).toEqual({
      type: 'image_url',
      image_url: { url: `data:image/png;base64,${TINY_PNG_B64}` },
    });
    expect(JSON.parse(JSON.stringify({ img }))).toEqual({
      img: {
        __gepa_image: {
          type: 'image_url',
          image_url: { url: `data:image/png;base64,${TINY_PNG_B64}` },
        },
      },
    });
  });

  it('reads path images and infers media type from extension', () => {
    const dir = mkdtempSync(`${tmpdir()}/gepa-ts-image-`);
    try {
      const png_path = join(dir, 'frame.png');
      writeFileSync(png_path, Buffer.from(TINY_PNG_B64, 'base64'));

      const part = new Image({ path: png_path }).to_openai_content_part();

      expect(part.type).toBe('image_url');
      expect(part.image_url.url).toBe(`data:image/png;base64,${TINY_PNG_B64}`);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('validates source combinations like upstream', () => {
    expect(() => new Image({})).toThrow('Exactly one of url, path, or base64_data must be provided.');
    expect(() => new Image({ url: 'x', base64_data: TINY_PNG_B64, media_type: 'image/png' })).toThrow(
      'Exactly one of url, path, or base64_data must be provided.',
    );
    expect(() => new Image({ base64_data: TINY_PNG_B64 })).toThrow(
      'media_type is required when using base64_data.',
    );
    expect(_guess_media_type('/tmp/image.unknown')).toBe('image/png');
  });

  it('flows through reflective prompt rendering as multimodal content', async () => {
    let observed_prompt: unknown = null;
    await InstructionProposalSignature.run(async (prompt) => {
      observed_prompt = prompt;
      return '```\nimproved\n```';
    }, {
      current_instruction_doc: 'inspect image',
      dataset_with_feedback: [
        {
          Feedback: {
            Rendered: new Image({ base64_data: TINY_PNG_B64, media_type: 'image/png' }),
          },
        },
      ],
      prompt_template: null,
    });

    expect(Array.isArray(observed_prompt)).toBe(true);
    const message = (observed_prompt as Array<{ content: unknown }>)[0];
    expect(Array.isArray(message?.content)).toBe(true);
    const content = message.content as Array<Record<string, unknown>>;
    expect(content[0]?.type).toBe('text');
    expect(String(content[0]?.text)).toContain('1 image(s)');
    expect(content[1]).toEqual({
      type: 'image_url',
      image_url: { url: `data:image/png;base64,${TINY_PNG_B64}` },
    });
  });
});
