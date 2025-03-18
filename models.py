from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, JSON, DECIMAL, SmallInteger, Date, \
    BigInteger, select
from sqlalchemy.orm import relationship, column_property

from core.database import Base  # Импортируйте Base из вашего файла database.py


class BookmarkType(Base):
    __tablename__ = 'bookmark_type'

    id = Column(BigInteger, primary_key=True)
    name = Column(String(128))
    is_default = Column(Integer)
    is_visible = Column(Integer)
    is_notify = Column(Integer)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=True)


class Bookmarks(Base):
    __tablename__ = 'bookmarks'

    id = Column(BigInteger, primary_key=True)
    # date = Column(DateTime) use currentDate
    bookmark_type_id = Column(BigInteger, ForeignKey('bookmark_type.id'), nullable=True)
    # !!!
    is_default = column_property(
        select(BookmarkType.is_default).where(BookmarkType.id == bookmark_type_id).scalar_subquery().label('is_default')
    )
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    user_id = Column(BigInteger, ForeignKey('users.id'))


class Categories(Base):
    __tablename__ = 'categories'

    id = Column(BigInteger, primary_key=True)
    name = Column(String(30), unique=True)
    description = Column(Text)
    dir = Column(String(120))


class CategoriesSites(Base):
    __tablename__ = 'categories_sites'

    id = Column(BigInteger, primary_key=True)
    category_id = Column(BigInteger, ForeignKey('categories.id'))
    site_id = Column(BigInteger, ForeignKey('django_site.id'))


class Collections(Base):
    __tablename__ = 'collections'

    id = Column(BigInteger, primary_key=True)
    name = Column(String(150))
    description = Column(Text)
    cover_source = Column(String(256), nullable=True)
    is_published = Column(Integer)
    position = Column(Integer)
    cover = Column(JSON)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=True)
    site_id = Column(BigInteger, ForeignKey('django_site.id'))
    count_comments = Column(Integer)
    score = Column(Integer)


class Comments(Base):
    __tablename__ = 'comments'

    id = Column(BigInteger, primary_key=True)
    text = Column(String(600))
    date = Column(DateTime)
    is_blocked = Column(Integer)
    is_deleted = Column(Integer)
    is_pinned = Column(Integer)
    is_spoiler = Column(Integer)
    title_id = Column(BigInteger, ForeignKey('titles.id'), nullable=True)
    user_id = Column(BigInteger, ForeignKey('users.id'))


class DjangoSite(Base):
    __tablename__ = 'django_site'

    id = Column(BigInteger, primary_key=True)
    domain = Column(String(100), unique=True)
    name = Column(String(50))


class Genres(Base):
    __tablename__ = 'genres'

    id = Column(BigInteger, primary_key=True)
    name = Column(String(30), unique=True)
    description = Column(Text)
    dir = Column(String(120))


class GenresSites(Base):
    __tablename__ = 'genres_sites'

    id = Column(BigInteger, primary_key=True)
    genre_id = Column(BigInteger, ForeignKey('genres.id'))
    site_id = Column(BigInteger, ForeignKey('django_site.id'))


class Payments(Base):
    __tablename__ = 'payments'

    id = Column(BigInteger, primary_key=True)
    type = Column(Integer)
    status = Column(Integer)
    date = Column(DateTime)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=True)


class Rating(Base):
    __tablename__ = 'rating'

    id = Column(BigInteger, primary_key=True)
    rating = Column(Integer)
    date = Column(DateTime, nullable=True)
    # is_deleted = Column(Integer)
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    user_id = Column(BigInteger, ForeignKey('users.id'))


class SimilarTitles(Base):
    __tablename__ = 'similar_titles'

    id = Column(BigInteger, primary_key=True)
    # стоит ли отрисовывать
    draw = Column(Integer, nullable=True)
    # совпадает жанр

    genre = Column(Integer, nullable=True)
    # совпадает история

    history = Column(Integer, nullable=True)
    # колво голосов

    score = Column(Integer)
    title1_id = Column(BigInteger, ForeignKey('titles.id'))
    title2_id = Column(BigInteger, ForeignKey('titles.id'))
    # создатель связи title1_id title2_id
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=True)


class SimilarTitlesVotes(Base):
    __tablename__ = 'similar_titles_votes'

    id = Column(BigInteger, primary_key=True)
    type = Column(SmallInteger)
    is_deleted = Column(Integer)
    similar_id = Column(BigInteger, ForeignKey('similar_titles.id'))
    user_id = Column(BigInteger, ForeignKey('users.id'))


class TagsTags(Base):
    __tablename__ = 'tags'

    id = Column(BigInteger, primary_key=True)
    name = Column(String(30), unique=True)
    description = Column(Text)
    dir = Column(String(120))


class TitlePromotion(Base):
    __tablename__ = 'title_promotion'

    id = Column(BigInteger, primary_key=True)
    is_active = Column(Integer)
    pause_delay = Column(Integer)
    date_start = Column(Date, nullable=True)
    date_end = Column(Date, nullable=True)
    days_left = Column(Integer)
    days_this_promo = Column(Integer)
    promo_history = Column(JSON)
    render_count = Column(Integer)
    render_count_this = Column(Integer)
    click_count = Column(Integer)
    click_count_this = Column(Integer)
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    render_count_day = Column(Integer)
    click_count_day = Column(Integer)


class TitleStatistics(Base):
    __tablename__ = 'title_statistics'

    id = Column(BigInteger, primary_key=True)
    date = Column(Date)
    votes = Column(BigInteger)
    f_votes = Column(BigInteger)
    views = Column(BigInteger)
    f_views = Column(BigInteger)
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    m_views = Column(BigInteger)
    m_votes = Column(BigInteger)


class TitleStatus(Base):
    __tablename__ = 'title_status'

    id = Column(BigInteger, primary_key=True)
    name = Column(String(30), unique=True)


class TitleStatusSites(Base):
    __tablename__ = 'title_status_sites'

    id = Column(BigInteger, primary_key=True)
    titlestatus_id = Column(BigInteger, ForeignKey('title_status.id'))
    site_id = Column(BigInteger, ForeignKey('django_site.id'))


class TitleType(Base):
    __tablename__ = 'title_type'

    id = Column(BigInteger, primary_key=True)
    name = Column(String(30), unique=True)
    description = Column(Text)
    dir = Column(String(120))


class TitleTypeSites(Base):
    __tablename__ = 'title_type_sites'

    id = Column(BigInteger, primary_key=True)
    titletype_id = Column(BigInteger, ForeignKey('title_type.id'))
    site_id = Column(BigInteger, ForeignKey('django_site.id'))


class Titles(Base):
    __tablename__ = 'titles'
    id = Column(BigInteger, primary_key=True)
    # features
    status_id = Column(BigInteger, ForeignKey('title_status.id'), nullable=True)
    age_limit = Column(Integer)
    count_chapters = Column(Integer)
    type_id = Column(BigInteger, ForeignKey('title_type.id'), nullable=True)
    # many_to_many_features
    # categories
    # genres
    cover = Column(JSON, default={})
    main_name = Column(String(200))
    dir = Column(String(200), unique=True)
    issue_year = Column(Integer, nullable=True)
    is_yaoi = Column(Integer)
    is_erotic = Column(Integer)
    upload_date = Column(DateTime)
    total_views = Column(Integer)
    total_votes = Column(Integer)
    avg_rating = Column(DECIMAL(10, 1))
    uploaded = Column(Integer)
    is_legal = Column(Integer)


class TitlesGenres(Base):
    __tablename__ = 'titles_genres'

    id = Column(BigInteger, primary_key=True)
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    genre_id = Column(BigInteger, ForeignKey('genres.id'))


class TitlesCategories(Base):
    __tablename__ = 'titles_categories'

    id = Column(BigInteger, primary_key=True)
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    category_id = Column(BigInteger, ForeignKey('categories.id'))


class TitlesCollections(Base):
    __tablename__ = 'titles_collections'

    id = Column(BigInteger, primary_key=True)
    pos = Column(Integer)
    collection_id = Column(BigInteger, ForeignKey('collections.id'))
    title_id = Column(BigInteger, ForeignKey('titles.id'))


class TitlesSites(Base):
    __tablename__ = 'titles_sites'

    id = Column(BigInteger, primary_key=True)
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    site_id = Column(BigInteger, ForeignKey('django_site.id'))


class TitleChapter(Base):
    __tablename__ = 'title_chapters'

    id = Column(BigInteger, primary_key=True)
    chapter = Column(String(30))
    is_published = Column(Boolean)
    is_deleted = Column(Boolean)
    title_id = Column(BigInteger, ForeignKey('titles.id'))


class UserBuys(Base):
    __tablename__ = 'user_buys'

    id = Column(BigInteger, primary_key=True)
    date = Column(DateTime)
    chapter_id = Column(BigInteger, ForeignKey('title_chapters.id'))
    payment_id = Column(BigInteger, ForeignKey('payments.id'), nullable=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=True)


class UserTitleData(Base):
    __tablename__ = 'user_title_data'

    id = Column(BigInteger, primary_key=True)
    # last_read_chapter = Column(BigInteger)
    last_read_date = Column(DateTime, nullable=True)
    title_id = Column(BigInteger, ForeignKey('titles.id'))
    user_id = Column(BigInteger, ForeignKey('users.id'))
    chapter_votes = Column(JSON, nullable=False)
    chapter_views = Column(JSON, nullable=False)


class RawUsers(Base):
    __tablename__ = 'users'

    id = Column(BigInteger, primary_key=True)
    last_login = Column(DateTime, nullable=True)
    is_superuser = Column(Integer)
    is_staff = Column(Integer)
    is_active = Column(Integer)
    date_joined = Column(DateTime)
    last_seen_date = Column(DateTime)
    yaoi = Column(SmallInteger)
    adult = Column(SmallInteger)
    preference = Column(Integer)
    is_banned = Column(Integer)
    birthday = Column(Date, nullable=True)
    sex = Column(Integer)
    is_premium = Column(Integer)


class TitlesTitleRelation(Base):
    __tablename__ = 'titles_titlerelation'

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True
    )
    type = Column(
        String(20),
        nullable=False
    )
    position = Column(
        SmallInteger(),
        nullable=False
    )
    title_id = Column(
        BigInteger,
        ForeignKey('titles.id'),
        nullable=False
    )
    relation_list_id = Column(
        BigInteger,
        ForeignKey('title_relations_lists.id'),
        nullable=True
    )
