Title: RSA Animate XBMC Plugin
Tagline: Watch videos from RSA Animate on XBMC
Slug: rsa-animate-xbmc-plugin
Date: 2012-05-20

</p>

<div class="intro">
Here's another mini XBMC plugin I've recently developed that allows you
to watch videos from <a href="http://comment.rsablogs.org.uk/videos/">RSA Animate</a> on XBMC.
</div>

</p>

### RSA

*The RSA (Royal Society for the encouragement of Arts, Manufactures and Commerce) describes itself as "an enlightenment organisation committed to finding innovative practical solutions to today's social challenges. Through its ideas, research and 27,000-strong Fellowship it seeks to understand and enhance human capability so we can close the gap between today's reality and people's hopes for a better world."*

If you haven't seen a talk animated by RSA, now is a good time to
[start][].

</p>

### The Plugin

</p>

The plugin itself is very simple. It utilises [BeautifulSoup][] to parse
the HTML scraped from the RSA website, and the video content is
collected with a couple of lines.

    :::python
    def scrape_site(contents):
        """Scrape the RSA Animate Video site and
        return an array of dictionaries
        """
        output = []
        soup = BeautifulSoup(str(contents))

        # All the H3s on the page appear to be video titles, let get the title string from each one
        posts = soup.findAll('div', 'post')

        for post in posts:
            # Clean titles
            title = post.h3.a.string
            # Remove the unneccessary prefix
            title = re.sub('RSA Animate', '', title)
            # Remove white space
            title = title.lstrip(' ')

            # Get the dates
            date = post.find('p', 'postmetadata').find('span', 'alignleft')

            # Get the Youtube URLs
            url = post.find('object').find('embed')['src']

            final_title = '{0} ({1})'.format(title, date.string)

            output.append(
                {'title' : final_title,
                 'url' : url})

            return output


I *love* you BeautifulSoup.

</p>

Then, I simply call the add the collection of directory items pointing
to the Youtube plugin.

    :::python
    video_list = scraper.scrape_site(contents)

    for video in video_list:
        xbmc_handler.add_video_link(video['title'], video['url'])

    #...

    def add_video_link(title, url):
        """Collect just the ID from a Youtube ID, and use it to return an XBMC Directory Item
        """
        # The Youtube ID is the 4th value with array split
        id = url.split('/')[4]

        # URL to call the Youtube plugin
        youtube_url = 'plugin://plugin.video.youtube?action=play_video&videoid=%s' % (id)

        # Create a new XBMC List Item and provide the title
        list_item = xbmcgui.ListItem(title)
        list_item.setProperty('IsPlayable', 'true')

        return xbmcplugin.addDirectoryItem(__addon_id_int__, youtube_url, list_item)

### Installation and Support

</p>

The plugin is not in the official repository just yet. I think, for it
to be worthy, I'll need to extend it to cover the wealth of video
content on the [RSA site][] beyond just the [Animate section][RSA
Animate]. But, it's simple to download and install via [Github][].

</p>

For support and discussion, please use the official [XBMC forum][].

</p>

  [start]: http://www.youtube.com/watch?feature=player_embedded&v=u6XAPnuFjJc
  [BeautifulSoup]: http://www.crummy.com/software/BeautifulSoup/
  [RSA site]: http://www.thersa.org
  [Github]: https://github.com/lextoumbourou/plugin.video.rsa_animate
  [XBMC forum]: http://forum.xbmc.org/showthread.php?tid=128173
