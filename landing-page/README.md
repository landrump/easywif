# EasyWIF Landing Page

A production-ready, single-page SaaS landing page built with Next.js, TypeScript, and Tailwind CSS.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Add your images to the `/public` folder:
   - `ezw logo_black.png`
   - `ezw logo_blue default.png`
   - `EasyWIF_Hero.png`
   - `EasyWIF_HowItWorks.png`
   - `EasyWIF_Project View.png`
   - `EasyWIF_Resource $K Impact.png`
   - `EasyWIF_Save and Export.png` (optional)

3. Update constants in `/app/constants.ts`:
   - `calendlyLink` - Your Calendly booking link
   - `contactEmail` - Your contact email
   - `linkedInUrl` - Your LinkedIn company URL

## Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Building for Production

Build the static site:

```bash
npm run build
```

The output will be in the `/out` folder, ready for deployment.

## Deployment to Netlify

### Option 1: Drag and Drop
1. Run `npm run build`
2. Drag the `/out` folder to [Netlify Drop](https://app.netlify.com/drop)

### Option 2: Git Integration
1. Push your code to GitHub/GitLab/Bitbucket
2. Connect your repository to Netlify
3. Netlify will automatically detect Next.js and build settings

### Option 3: Netlify CLI
```bash
npm install -g netlify-cli
netlify deploy --prod
```

## Custom Domain Setup

1. In Netlify dashboard, go to Site settings → Domain management
2. Add your custom domain
3. Follow Netlify's DNS instructions to update your domain registrar
4. Netlify will automatically provision SSL certificates

## Image Notes

- All images should be placed in the `/public` folder
- If your images have different extensions (.jpg instead of .png), update the image paths in the components
- Use optimized images for best performance (recommended: WebP format)

## Customization

- **Copy/Text**: Edit the component files in `/components/`
- **Colors**: Update `tailwind.config.js` to match your brand
- **Styling**: All components use Tailwind CSS classes
- **Video**: Add your Loom/YouTube embed in `/components/VideoSection.tsx`

## Netlify Forms

The demo request form is configured for Netlify Forms. Make sure to:
1. Add `data-netlify="true"` attribute (already included)
2. Add a hidden input with `name="form-name"` (already included)
3. Netlify will automatically handle form submissions

## Support

For questions or issues, contact the development team.

