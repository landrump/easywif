# Deployment Guide for EasyWIF Landing Page

## Quick Start - Netlify Deployment

### Step 1: Prepare Your Images

1. Copy all your images to the `/landing-page/public/` folder:
   - `ezw logo_black.png`
   - `ezw logo_blue default.png`
   - `EasyWIF_Hero.png`
   - `EasyWIF_HowItWorks.png`
   - `EasyWIF_Project View.png`
   - `EasyWIF_Resource $K Impact.png`
   - `EasyWIF_Save and Export.png` (optional)

2. **Important**: If your images have different file extensions (.jpg, .webp, etc.), update the image paths in these files:
   - `components/Navigation.tsx` (logo)
   - `components/Hero.tsx`
   - `components/HowItWorks.tsx`
   - `components/Product.tsx`
   - `components/Footer.tsx` (logo)

### Step 2: Update Configuration

1. Edit `/app/constants.ts` and update:
   - `calendlyLink`: Your Calendly booking URL
   - `contactEmail`: Your contact email address
   - `linkedInUrl`: Your LinkedIn company page URL

### Step 3: Install Dependencies

```bash
cd landing-page
npm install
```

### Step 4: Test Locally (Optional)

```bash
npm run dev
```

Visit http://localhost:3000 to preview your site.

### Step 5: Build for Production

```bash
npm run build
```

This creates a static site in the `/out` folder.

## Deploying to Netlify

### Method 1: Drag and Drop (Easiest)

1. After running `npm run build`, you'll have an `/out` folder
2. Go to [Netlify Drop](https://app.netlify.com/drop)
3. Drag the entire `/out` folder onto the page
4. Your site will be live immediately with a netlify.app URL

### Method 2: Git Integration (Recommended for Updates)

1. Push your code to GitHub/GitLab/Bitbucket
2. Sign in to [Netlify](https://app.netlify.com)
3. Click "Add new site" → "Import an existing project"
4. Connect your Git provider and select your repository
5. Netlify will auto-detect Next.js settings:
   - **Build command**: `npm run build`
   - **Publish directory**: `out`
6. Click "Deploy site"

### Method 3: Netlify CLI

```bash
# Install Netlify CLI globally
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy (first time)
netlify deploy --prod

# Future deployments
netlify deploy --prod
```

## Connecting Your Custom Domain

1. In Netlify dashboard, go to your site
2. Navigate to **Site settings** → **Domain management**
3. Click **Add custom domain**
4. Enter your domain name (e.g., `easywif.com`)
5. Netlify will show you DNS records to add:
   - **A Record**: Point to Netlify's IP
   - **CNAME Record**: Point to your Netlify site URL
6. Update DNS at your domain registrar:
   - Go to your domain registrar (GoDaddy, Namecheap, etc.)
   - Find DNS settings
   - Add the records Netlify provides
7. Wait for DNS propagation (usually 5-60 minutes)
8. Netlify will automatically provision SSL certificate (HTTPS)

## Setting Up Netlify Forms

The demo request form is already configured for Netlify Forms:

1. The form in `components/BookDemo.tsx` has `data-netlify="true"`
2. Netlify will automatically detect and handle form submissions
3. View submissions in Netlify dashboard → **Forms**
4. Set up email notifications in **Site settings** → **Forms** → **Form notifications**

## Adding Your Video

1. Get your Loom or YouTube embed URL
2. Open `components/VideoSection.tsx`
3. Replace the placeholder div with the iframe code (instructions in comments)
4. Example:
```tsx
<div className="aspect-video rounded-lg overflow-hidden shadow-lg">
  <iframe
    src="https://www.loom.com/embed/YOUR_VIDEO_ID"
    className="w-full h-full"
    allowFullScreen
  ></iframe>
</div>
```

## Troubleshooting

### Images Not Showing
- Check file names match exactly (case-sensitive)
- Verify images are in `/public` folder
- Check file extensions match the code (.png vs .jpg)

### Build Errors
- Make sure all dependencies are installed: `npm install`
- Check Node.js version (requires Node 18+)
- Review error messages in terminal

### Domain Not Working
- Verify DNS records are correct
- Wait for DNS propagation (can take up to 48 hours)
- Check Netlify's domain status in dashboard

### Forms Not Submitting
- Ensure `data-netlify="true"` is on the form
- Check Netlify Forms are enabled in site settings
- Verify form has a `name` attribute

## Next Steps

- Customize colors in `tailwind.config.js`
- Update copy in component files
- Add analytics (Google Analytics, Plausible, etc.)
- Set up custom 404 page if needed
- Configure redirects in `netlify.toml` if needed

## Support

For Netlify-specific issues, check [Netlify Docs](https://docs.netlify.com/)
For Next.js issues, check [Next.js Docs](https://nextjs.org/docs)

